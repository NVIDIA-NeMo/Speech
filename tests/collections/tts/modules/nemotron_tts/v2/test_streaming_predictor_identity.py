# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch

from nemo.collections.tts.models.nemotron_tts.v2.base import NemotronTTSBase
from tests.collections.tts.modules.nemotron_tts.v2.test_streaming_timestamps_model import _init_stream, _model

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('use_cfg', [False, True])
def test_timestamps_preserve_actual_acoustic_predictor_sampling_and_cache_progress(use_cfg):
    baseline, observed = _model(), _model(enabled=True)
    for model in (baseline, observed):
        # The timing fixture fixes audio codes to isolate clock/index assertions.
        # Restore the actual V2 method here, including every predictor block and
        # its sampling path. Only codec and speaker-context I/O remain stubbed.
        del model._predict_audio_codes
        assert model._predict_audio_codes.__func__ is NemotronTTSBase._predict_audio_codes
        with torch.no_grad():
            for block in model.acoustic_codes_predictor.blocks:
                bias = block.codebook_projection.bias.view(block.num_codes, block.num_output_tokens)
                bias[:, -1].fill_(-10000.0)
        assert not hasattr(model, 'text_aligner') and not hasattr(model, 'aligned_text_head')
    states = [_init_stream(model, use_cfg=use_cfg) for model in (baseline, observed)]
    for state in states:
        state.config.temperature = 0.7
        state.config.topk = 80
    first_audio_step = first_event_step = first_frame_step = None
    streamed_frames = []
    previous_progress = [cache.get_seq_length() for cache in states[0].acoustic_codes_predictor_cache]
    tokens = [7, 8, 9, 10, 11, 12, baseline.eos_id, baseline.eos_id] + [None] * 5
    with patch(
        'nemo.collections.tts.modules.glow_tts_aligner.TinyGlowTTSAligner.forward',
        side_effect=AssertionError('Timestamp recording cannot call the acoustic teacher'),
    ):
        for step, token_id in enumerate(tokens):
            token = None if token_id is None else torch.full((2,), token_id, dtype=torch.long)
            rng_before = _rng_snapshot()
            ordinary = baseline.streaming_step(states[0], token)
            rng_after = _rng_snapshot()
            _restore_rng(rng_before)
            captured = observed.streaming_step(states[1], token)
            _assert_rng_equal(rng_after, _rng_snapshot())
            states = [ordinary[0], captured[0]]
            torch.testing.assert_close(states[0].last_hidden, states[1].last_hidden, rtol=0, atol=0)
            assert (ordinary[1] is None) == (captured[1] is None)
            if ordinary[1] is not None:
                torch.testing.assert_close(ordinary[1], captured[1], rtol=0, atol=0)
                assert not ordinary[1].eq(baseline.audio_eos_id).any()
                first_audio_step = step if first_audio_step is None else first_audio_step
            progress = []
            for left, right in zip(states[0].acoustic_codes_predictor_cache, states[1].acoustic_codes_predictor_cache):
                assert left.get_seq_length() == right.get_seq_length() == states[0].cache_seq_len
                progress.append(left.get_seq_length())
            assert progress == [length + 1 for length in previous_progress]
            previous_progress = progress
            events = observed.drain_timestamp_events(states[1])
            if events:
                first_event_step = step if first_event_step is None else first_event_step
                assert all(not event.get('forced') or event['confidence'] == 0.0 for event in events)
            frames = observed.drain_timestamp_frames(states[1])
            if frames:
                first_frame_step = step if first_frame_step is None else first_frame_step
                assert captured[1] is not None
                assert len(frames) == 2  # One current-query record per conditional batch row.
                assert {frame['batch_index'] for frame in frames} == {0, 1}
                assert all(frame['valid_codec_frames'] == observed.frame_stacking_factor for frame in frames)
                assert all(not frame['forced'] or frame['confidence'] == 0.0 for frame in frames)
                streamed_frames.extend(frames)
            else:
                assert captured[1] is None
            assert baseline.drain_timestamp_events(states[0]) == []
            assert baseline.drain_timestamp_frames(states[0]) == []
            assert not states[0].finished.any() and not states[1].finished.any()
    assert first_audio_step is not None and first_event_step == first_audio_step == first_frame_step
    snapshots = observed.get_timestamps(states[1])
    for row, snapshot in enumerate(snapshots):
        assert snapshot['frames'] == [frame for frame in streamed_frames if frame['batch_index'] == row]
        assert snapshot['frame_count'] == len(snapshot['frames'])
        assert [frame['frame_index'] for frame in snapshot['frames']] == list(range(snapshot['frame_count']))


def test_target_text_warmup_retains_keys_without_emitting_timestamps_for_silence():
    model = _model(enabled=True)
    state = _init_stream(model, use_cfg=True)
    for token_id in [7, 8, 9, 10, 11, 12]:
        state, _, _ = model.streaming_step(state, torch.full((2,), token_id))
    assert model.drain_timestamp_events(state)
    previous_turn = model.get_timestamps(state)[0]['turns'][-1]['turn_index']
    model.reset_timestamp_turn(state)
    model.drain_timestamp_events(state)
    with torch.inference_mode():
        state.turn_text_tokens_seen.zero_()
        state.text_finished.zero_()
        state.audio_steps.zero_()
    for token_id in [13, 14, 15]:
        state, codes, _ = model.streaming_step(
            state, torch.full((2,), token_id), prefill_like_step=True, timestamp_text_warmup=True
        )
        assert codes is None and model.drain_timestamp_events(state) == []
    current = model.get_timestamps(state)[0]
    turn = current['turns'][-1]
    assert turn['turn_index'] == previous_turn + 1
    assert turn['source_token_ids'] == [13, 14, 15]
    assert all(
        record['start'] is None for record in current['timestamps'] if record['turn_index'] == turn['turn_index']
    )
    for token_id in [16, 17]:
        state, codes, _ = model.streaming_step(state, torch.full((2,), token_id))
        assert codes is None and model.drain_timestamp_events(state) == []
    state, codes, _ = model.streaming_step(state, None)
    events = model.drain_timestamp_events(state)
    assert codes is not None and events
    assert all(event['turn_index'] == turn['turn_index'] for event in events)
    assert model.get_timestamps(state)[0]['turns'][-1]['source_token_ids'] == [13, 14, 15, 16, 17]


def _rng_snapshot():
    return torch.random.get_rng_state().clone(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []


def _restore_rng(snapshot):
    torch.random.set_rng_state(snapshot[0])
    if snapshot[1]:
        torch.cuda.set_rng_state_all(snapshot[1])


def _assert_rng_equal(expected, actual):
    assert torch.equal(expected[0], actual[0])
    assert len(expected[1]) == len(actual[1])
    assert all(torch.equal(left, right) for left, right in zip(expected[1], actual[1]))
