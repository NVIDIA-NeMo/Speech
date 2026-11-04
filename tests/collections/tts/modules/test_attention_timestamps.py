# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json

import pytest

from nemo.collections.tts.modules.attention_timestamps import AttentionTimestampTracker, StreamingAttentionPathDecoder

pytestmark = pytest.mark.unit


def test_streaming_tokens_do_not_require_a_complete_future_transcript():
    tracker = AttentionTimestampTracker()
    tracker.register_token(0, 17, 'hello')
    events = tracker.update([0.8], start=0.0, end=0.08)
    assert events[0]['event'] == 'start' and events[0]['status'] == 'provisional'
    tracker.register_token(1, 18, 'world')
    tracker.update([0.7, 0.1], start=0.08, end=0.16)
    confirmed = tracker.drain_events()[-1]
    assert confirmed['event'] == 'confirmed'
    assert confirmed['start'] == 0.0
    assert confirmed['emitted_at'] == 0.16 and confirmed['latency_seconds'] == 0.16
    tracker.update([0.1, 0.7], start=0.16, end=0.24)
    tracker.update([0.1, 0.8], start=0.24, end=0.32)
    records = tracker.finalize(0.32)
    assert [(token['start'], token['end']) for token in records] == [(0.0, 0.16), (0.16, 0.32)]
    assert all(token['status'] == 'finalized' for token in records)
    assert tracker.drain_events() and tracker.drain_events() == []


def test_repeated_token_ids_keep_distinct_input_positions_and_one_frame_forward_tokens():
    tracker = _tracker([31, 31, 31, 9])
    for index in range(4):
        tracker.update(_one_hot(index, 4), start=index * 0.08, end=(index + 1) * 0.08)
    records = tracker.finalize(0.32)
    assert [token['token_id'] for token in records] == [31, 31, 31, 9]
    assert [token['token_index'] for token in records] == [0, 1, 2, 3]
    assert [value for token in records for value in (token['start'], token['end'])] == pytest.approx(
        [0.0, 0.08, 0.08, 0.16, 0.16, 0.24, 0.24, 0.32]
    )
    assert all(token['short_evidence'] and not token['spans'][0]['confirmed'] for token in records)


def test_realistic_mix_with_thirty_percent_one_frame_tokens_does_not_erase_short_bpes():
    tracker = _tracker(list(range(10)))
    durations = [1, 3, 2, 1, 2, 3, 1, 4, 2, 3]
    clock = 0.0
    for token_index, frames in enumerate(durations):
        for _ in range(frames):
            tracker.update(_one_hot(token_index, 10), start=clock, end=clock + 0.04)
            clock += 0.04
    records = tracker.finalize(clock)
    assert all(token['start'] is not None for token in records)
    assert sum(token['short_evidence'] for token in records) == 3
    assert [token['spans'][0]['frame_count'] for token in records] == durations


def test_single_frame_excursion_is_retracted_without_filling_the_gap_with_the_previous_token():
    tracker = _tracker([10, 11])
    for index, token in enumerate([0, 0, 1, 0, 0]):
        tracker.update(_one_hot(token, 2), start=index * 0.1, end=(index + 1) * 0.1)
    records = tracker.finalize(0.5)
    assert records[1]['status'] == 'unaligned' and records[1]['start'] is None
    assert [value for span in records[0]['spans'] for value in (span['start'], span['end'])] == pytest.approx(
        [0.0, 0.2, 0.3, 0.5]
    )
    assert records[0]['ambiguous']
    assert records[0]['start'] == 0.0 and records[0]['end'] == 0.2
    assert any(event['event'] == 'retract' and event['token_index'] == 1 for event in tracker.drain_events())


def test_confirmed_backward_revisit_has_disjoint_spans_and_a_primary_contiguous_interval():
    tracker = _tracker([10, 11])
    for index, token in enumerate([0, 0, 1, 1, 0, 0, 0]):
        tracker.update(_one_hot(token, 2), start=index * 0.1, end=(index + 1) * 0.1)
    records = tracker.finalize(0.7)
    assert [value for span in records[0]['spans'] for value in (span['start'], span['end'])] == pytest.approx(
        [0.0, 0.2, 0.4, 0.7]
    )
    assert records[0]['start'] == 0.4 and records[0]['end'] == pytest.approx(0.7)
    assert records[0]['primary_span_index'] == 1 and records[0]['ambiguous']
    assert records[1]['start'] == 0.2 and records[1]['end'] == 0.4


@pytest.mark.parametrize('values', [[0.0, 0.0], [float('nan'), 0.9], [float('inf'), 0.0], [-0.1, 0.8], [0.8, 0.8]])
def test_invalid_and_zero_probability_frames_never_create_token_evidence(values):
    tracker = _tracker([10, 11])
    tracker.update(values, start=0.0, end=0.1)
    records = tracker.finalize(0.1)
    assert all(token['start'] is None and token['spans'] == [] for token in records)
    json.dumps(records, allow_nan=False)


@pytest.mark.parametrize(
    'settings,values',
    [
        ({'min_confidence': 0.6}, [0.4, 0.1]),
        ({'min_text_mass': 0.6}, [0.4, 0.1]),
        ({'min_margin': 0.4}, [0.5, 0.3]),
    ],
)
def test_confidence_gates_use_original_full_key_probabilities(settings, values):
    tracker = _tracker([10, 11], **settings)
    tracker.update(values, start=0.0, end=0.1)
    records = tracker.finalize(0.1)
    assert all(token['status'] == 'unaligned' for token in records)


def test_audio_gaps_and_low_evidence_do_not_smear_token_duration():
    tracker = _tracker([10], confirmation_frames=1)
    tracker.update([0.7], start=0.5, end=0.6)
    tracker.update([0.0], start=0.6, end=0.7)
    tracker.update([0.8], start=0.9, end=1.0)
    record = tracker.finalize(2.0)[0]
    assert [(span['start'], span['end']) for span in record['spans']] == [(0.5, 0.6), (0.9, 1.0)]
    assert record['start'] == 0.9 and record['end'] == 1.0


def test_unregistered_future_positions_are_not_given_timestamps():
    tracker = _tracker([10])
    tracker.update([0.9], token_indices=[4], start=0.0, end=0.1)
    assert tracker.finalize(0.1)[0]['start'] is None


def test_nonwinning_positive_attention_is_reported_without_fabricating_zero_duration():
    tracker = _tracker([10, 11, 12], confirmation_frames=1)
    tracker.update([0.5, 0.3, 0.0], start=0.0, end=0.1)
    records = tracker.finalize(0.1)
    assert records[1]['start'] is None and records[1]['end'] is None
    assert records[1]['reason'] == 'never_selected'
    assert records[1]['posterior_support'] == pytest.approx(0.03)
    assert records[1]['max_probability'] == 0.3
    assert records[2]['reason'] == 'no_positive_attention'


def test_explicit_key_indices_and_batched_probability_rows_have_identical_results_to_stream_registration():
    streamed = AttentionTimestampTracker(confirmation_frames=1)
    registered = AttentionTimestampTracker(confirmation_frames=1)
    for index in [4, 9, 15]:
        registered.register_token(index, index + 10)
    for step, index in enumerate([4, 9, 15]):
        streamed.register_token(index, index + 10)
        streamed.update([0.8], token_indices=[index], start=step * 0.1, end=(step + 1) * 0.1)
        registered.update([0.8], token_indices=[index], start=step * 0.1, end=(step + 1) * 0.1)
    assert streamed.finalize(0.3) == registered.finalize(0.3)


def test_chunked_updates_and_single_frame_updates_have_identical_boundaries_and_events():
    single, chunked = _tracker([10, 11, 12]), _tracker([10, 11, 12])
    rows = [_one_hot(index, 3) for index in (0, 0, 1, 2, 2)]
    starts, ends = [i * 0.08 for i in range(5)], [(i + 1) * 0.08 for i in range(5)]
    for row, start, end in zip(rows, starts, ends):
        single.update(row, start=start, end=end)
    chunked.update_many(rows[:2], starts=starts[:2], ends=ends[:2])
    chunked.update_many(rows[2:], starts=starts[2:], ends=ends[2:])
    assert single.finalize(0.4) == chunked.finalize(0.4)
    assert single.drain_events() == chunked.drain_events()


def test_partial_last_frame_and_short_final_token_are_retained_without_extending_to_padding():
    tracker = _tracker([10, 11])
    tracker.update([0.8, 0.1], start=0.0, end=0.08)
    tracker.update([0.8, 0.1], start=0.08, end=0.16)
    tracker.update([0.1, 0.8], start=0.16, end=0.19)
    records = tracker.finalize(0.25)
    assert records[1]['start'] == 0.16 and records[1]['end'] == 0.19
    assert records[1]['short_evidence']


def test_zero_frame_eos_and_unreceived_tail_have_null_timestamps():
    tracker = _tracker([10, 11])
    assert tracker.update([0.8, 0.1], start=0.0, end=0.0) == []
    records = tracker.finalize(0.0)
    assert all(token['status'] == 'unaligned' and token['start'] is None for token in records)


def test_snapshots_and_events_are_independent_and_finalized_span_events_do_not_change():
    tracker = _tracker([10, 11], confirmation_frames=1)
    tracker.update([0.7, 0.1], start=0.0, end=0.1)
    snapshot = tracker.snapshot()
    snapshot[0]['spans'][0]['end'] = 100
    tracker.update([0.1, 0.7], start=0.1, end=0.2)
    events = tracker.drain_events()
    first_end = copy.deepcopy(next(event for event in events if event['event'] == 'end'))
    tracker.update([0.7, 0.1], start=0.2, end=0.3)
    assert next(event for event in events if event['event'] == 'end') == first_end
    assert tracker.finalize(0.3)[0]['spans'][0]['end'] == 0.1


def test_finalization_is_idempotent_and_rejects_late_mutation_or_retroactive_clipping():
    tracker = _tracker([10], confirmation_frames=1)
    tracker.update([0.8], start=0.0, end=0.1)
    with pytest.raises(ValueError, match='clip before update'):
        tracker.finalize(0.05)
    first = tracker.finalize(0.1)
    assert tracker.finalize(0.1) == first
    with pytest.raises(ValueError, match='change duration'):
        tracker.finalize(0.2)
    with pytest.raises(ValueError, match='already finalized'):
        tracker.update([0.8], start=0.1, end=0.2)
    with pytest.raises(ValueError, match='already finalized'):
        tracker.register_token(1, 11)


@pytest.mark.parametrize(
    'settings',
    [
        {'confirmation_frames': 0},
        {'confirmation_frames': True},
        {'confirmation_frames': 1.5},
        {'min_confidence': -0.1},
        {'min_confidence': float('nan')},
        {'min_text_mass': 1.1},
        {'min_margin': float('inf')},
        {'min_margin': True},
    ],
)
def test_invalid_tracker_settings_fail_early(settings):
    with pytest.raises(ValueError):
        AttentionTimestampTracker(**settings)


def test_malformed_key_mapping_and_overlapping_audio_intervals_fail_instead_of_misaligning():
    tracker = _tracker([10, 11])
    with pytest.raises(ValueError, match='matching lengths'):
        tracker.update([0.8], start=0.0, end=0.1)
    with pytest.raises(ValueError, match='distinct'):
        tracker.update([0.5, 0.4], token_indices=[0, 0], start=0.0, end=0.1)
    tracker.update([0.8, 0.1], start=0.0, end=0.1)
    with pytest.raises(ValueError, match='ordered'):
        tracker.update([0.8, 0.1], start=0.09, end=0.2)
    with pytest.raises(ValueError, match='identity'):
        tracker.register_token(0, 99)


def test_causal_force_progress_escapes_a_skip_penalty_stall_and_emits_estimates_immediately():
    ordinary = _tracker([17, 17, 9, 20], method='monotonic', force_progress=False)
    tracker = _tracker([17, 17, 9, 20], method='monotonic')
    rows = [[0.9, 0.08, 0.02, 0], [0.1, 0.04, 0.8, 0.06], [0.01, 0.01, 0.08, 0.9]]
    for frame, row in enumerate(rows):
        start, end = frame * 0.08, (frame + 1) * 0.08
        ordinary.update(row, start=start, end=end)
        events = tracker.update(row, start=start, end=end)
        assert events and all(event['emitted_at'] == pytest.approx(end) for event in events)
        if frame == 1:
            estimates = [event for event in events if event['forced']]
            assert {event['token_index'] for event in estimates} == {1, 2}
            assert all(event['confidence'] == 0 for event in estimates)
            assert all(event['start'] >= start and event['end'] <= end for event in estimates)
            assert all(event['attention_confidence'] > 0 for event in estimates)
    records = tracker.finalize(0.24)
    assert all(record['start'] is not None for record in records)
    assert records[1]['start'] == pytest.approx(0.08) and records[1]['end'] == pytest.approx(0.12)
    assert records[2]['start'] == pytest.approx(0.12) and records[2]['end'] == pytest.approx(0.16)
    assert records[1]['forced'] and records[1]['confidence'] == 0
    assert records[2]['forced'] and records[2]['confidence'] == 0
    assert records[3]['confidence'] == pytest.approx(0.9) and not records[3]['forced']
    assert ordinary.finalize(0.24)[3]['start'] is None


def test_monotonic_zero_evidence_skip_and_unspoken_punctuation_remain_null():
    tracker = _tracker([10, 11, 12, 13], method='monotonic')
    rows = [[0.9, 0, 0, 0], [0.1, 0, 0.8, 0], [0, 0, 0.9, 0]]
    for frame, row in enumerate(rows):
        tracker.update(row, start=frame * 0.08, end=(frame + 1) * 0.08)
    records = tracker.finalize(0.24)
    assert records[0]['start'] == 0.0 and records[2]['end'] == pytest.approx(0.24)
    assert all(records[index]['start'] is None and records[index]['end'] is None for index in (1, 3))
    assert all(records[index]['reason'] == 'no_positive_attention' for index in (1, 3))


def test_monotonic_streaming_visibility_and_noncontiguous_decoder_keys_keep_positions():
    tracker = AttentionTimestampTracker(method='monotonic')
    tracker.register_token(67, 17)
    tracker.update([0.8, 0.1], token_indices=[67, 90], start=0.0, end=0.08)
    tracker.register_token(90, 17)
    tracker.update([0.1, 0.8], token_indices=[67, 90], start=0.08, end=0.16)
    tracker.update([0.05, 0.85], token_indices=[67, 90], start=0.16, end=0.24)
    records = tracker.finalize(0.24)
    assert [record['token_index'] for record in records] == [67, 90]
    assert records[0]['start'] == 0 and records[0]['end'] == 0.08
    assert records[1]['start'] == 0.08 and records[1]['observed_frames'] == 2


def test_monotonic_chunks_preserve_events_and_boundaries_and_do_not_move_committed_prefix():
    rows = [[0.8, 0.19, 0.01], [0.51, 0.48, 0.01], [0.01, 0.19, 0.8], [0.01, 0.1, 0.89]]
    rows += [[0.99, 0, 0.01]] * 6  # Later evidence cannot revise a committed earlier decision.
    single = _tracker([17, 18, 19], method='monotonic')
    chunked = _tracker([17, 18, 19], method='monotonic')
    starts, ends = [i * 0.08 for i in range(len(rows))], [(i + 1) * 0.08 for i in range(len(rows))]
    for row, start, end in zip(rows, starts, ends):
        single.update(row, start=start, end=end)
    chunked.update_many(rows[:3], starts=starts[:3], ends=ends[:3])
    chunked.update_many(rows[3:], starts=starts[3:], ends=ends[3:])
    assert single.finalize(ends[-1]) == chunked.finalize(ends[-1])
    assert single.drain_events() == chunked.drain_events()
    records = single.snapshot()
    assert records[0]['start'] == 0 and records[0]['end'] == 0.16
    assert not any(record['ambiguous'] for record in records)


def test_causal_decoder_emits_every_frame_without_retaining_attention_history():
    decoder = StreamingAttentionPathDecoder()
    for frame in range(5000):
        decisions = decoder.update([0.1, 0.8, 0.1], start=frame * 0.08, end=(frame + 1) * 0.08)
        assert len(decisions) == 1
        assert decisions[0]['emitted_at'] == pytest.approx((frame + 1) * 0.08)
        retained = [value for value in vars(decoder).values() if isinstance(value, (dict, list, set, tuple))]
        assert all(len(value) <= 3 for value in retained)
    decisions = decoder.finalize(400.0)
    assert decisions == []


def test_monotonic_invalid_frames_and_real_audio_gaps_never_fill_in_missing_audio():
    tracker = _tracker([10], method='monotonic')
    for row, start, end in (([0.8], 0, 0.08), ([float('nan')], 0.08, 0.16), ([0.8], 0.5, 0.58)):
        tracker.update(row, start=start, end=end)
    record = tracker.finalize(0.7)[0]
    assert [(span['start'], span['end']) for span in record['spans']] == [(0.0, 0.08), (0.5, 0.58)]
    assert record['ambiguous']


def test_forced_confidence_stays_zero_when_later_frames_naturally_support_the_same_token():
    tracker = _tracker([10, 11, 12], method='monotonic', confirmation_frames=1)
    tracker.update([0.9, 0.09, 0.01], start=0, end=0.08)
    tracker.update([0.1, 0.05, 0.85], start=0.08, end=0.16)
    events = tracker.update([0.01, 0.01, 0.98], start=0.16, end=0.24)
    record = tracker.snapshot()[2]
    assert record['forced'] and record['confidence'] == 0
    assert record['attention_confidence'] > 0.9
    assert record['spans'][0]['forced'] and record['spans'][0]['confidence'] == 0
    assert all(event['forced'] and event['confidence'] == 0 for event in events)


def test_adjacent_natural_progress_retains_attention_probability_and_does_not_force():
    tracker = _tracker([10, 11, 12], method='monotonic')
    for index in range(3):
        events = tracker.update(_one_hot(index, 3), start=index * 0.08, end=(index + 1) * 0.08)
        assert all(not event['forced'] and event['confidence'] == pytest.approx(0.9) for event in events)
    records = tracker.finalize(0.24)
    assert all(not token['forced'] and token['confidence'] == token['attention_confidence'] for token in records)


def test_forward_forcing_uses_relative_text_support_but_stops_when_text_attention_collapses():
    tracker = _tracker([10, 11, 12], method='monotonic')
    tracker.update([0.9, 0.09, 0.01], start=0, end=0.08)
    tracker.update([0.001, 0.0005, 0.0085], start=0.08, end=0.16)
    assert not any(token['forced'] for token in tracker.snapshot())
    tracker.update([0.1, 0.05, 0.85], start=0.16, end=0.24)
    assert tracker.snapshot()[1]['forced']


def test_weak_diffuse_or_backward_attention_does_not_force_forward_progress():
    tracker = _tracker([10, 11, 12, 13], method='monotonic')
    tracker.update([0, 0.85, 0.10, 0.05], start=0, end=0.08)
    # Current support faded, but no forward position reaches half its peak.
    tracker.update([0.4, 0.1, 0.26, 0.24], start=0.08, end=0.16)
    assert not any(token['forced'] for token in tracker.snapshot())
    tracker.update([0.99, 0.01, 0, 0], start=0.16, end=0.24)
    assert tracker.snapshot()[3]['start'] is None


def test_collapsed_text_mass_does_not_impute_crossed_tokens_on_a_natural_skip():
    tracker = _tracker([10, 11, 12], method='monotonic')
    tracker.update([0.9, 0.09, 0.01], start=0, end=0.08)
    tracker.update([0, 0.000001, 0.000999], start=0.08, end=0.16)
    records = tracker.snapshot()
    assert records[1]['start'] is None
    assert not any(record['forced'] for record in records)


def test_forced_forward_progress_never_assigns_zero_support_positions_or_unobserved_tail():
    tracker = _tracker([10, 11, 12, 13, 14], method='monotonic')
    tracker.update([0.9, 0.05, 0.04, 0.01, 0], start=0, end=0.08)
    events = tracker.update([0.1, 0, 0.05, 0.85, 0], start=0.08, end=0.16)
    assert {event['token_index'] for event in events if event['forced']} == {2, 3}
    records = tracker.finalize(0.5)
    assert records[1]['start'] is None and records[4]['start'] is None
    assert records[3]['end'] == pytest.approx(0.16)
    assert not any(event['event'] == 'start' for event in tracker.drain_events() if event['emitted_at'] == 0.5)


def test_forced_partial_frame_subdivision_does_not_revise_previously_published_boundaries():
    tracker = _tracker([10, 11, 12], method='monotonic')
    tracker.update([0.9, 0.09, 0.01], start=0, end=0.08)
    before = copy.deepcopy(tracker.snapshot()[0])
    events = tracker.update([0.1, 0.05, 0.85], start=0.08, end=0.10)
    assert tracker.snapshot()[0]['start'] == before['start']
    assert tracker.snapshot()[0]['end'] == before['end']
    forced = [event for event in events if event['forced']]
    assert forced and all(0.08 <= event['start'] < event['end'] <= 0.10 for event in forced)
    assert all(event['emitted_at'] == 0.10 for event in events)


@pytest.mark.parametrize(
    'settings',
    [
        {'method': 'unknown'},
        {'force_progress': 1},
        {'progress_threshold': -1},
        {'progress_threshold': True},
        {'progress_threshold': float('nan')},
        {'skip_penalty': -1},
        {'skip_penalty': float('nan')},
        {'skip_penalty': True},
    ],
)
def test_invalid_monotonic_settings_fail_early(settings):
    with pytest.raises(ValueError):
        AttentionTimestampTracker(**settings)


def _tracker(token_ids, **settings):
    tracker = AttentionTimestampTracker(**settings)
    for index, token_id in enumerate(token_ids):
        tracker.register_token(index, token_id)
    return tracker


def _one_hot(index, size):
    return [0.9 if position == index else 0.0 for position in range(size)]
