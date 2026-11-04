# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from types import SimpleNamespace

import pytest

from nemo.collections.tts.modules.magpietts_inference.token_timestamps import (
    build_token_timestamps,
    save_token_timestamps,
)

pytestmark = pytest.mark.unit


def test_input_occurrences_keep_positions_across_mid_end_repetitions_and_unaligned_punctuation():
    trace = _trace()
    before = copy.deepcopy(trace)
    result = build_token_timestamps(
        trace, _tokenizer(), expected_input_text='a b a . a', expected_input_token_ids=[1, 2, 1, 3, 1]
    )
    tokens = result['timestamps']
    assert [token['token_index'] for token in tokens] == [0, 1, 2, 3, 4]
    assert [token['token_id'] for token in tokens] == [1, 2, 1, 3, 1]
    assert [token['start'] for token in tokens] == [0.0, 0.08, 0.24, None, None]
    assert [token['end'] for token in tokens] == [0.08, 0.24, 0.32, None, None]
    assert tokens[2]['token'] == 'a' and tokens[3]['token'] == '.'
    assert tokens[3]['status'] == 'unaligned' and tokens[3]['observed']
    assert tokens[4]['status'] == 'unobserved' and not tokens[4]['observed']
    assert result['expected_input_text'] == 'a b a . a'
    assert trace == before


def test_primary_timestamp_does_not_join_disjoint_visits_into_a_false_long_interval():
    trace = _trace()
    trace['timestamps'][0]['spans'].append(_span(0.48, 0.64, confidence=0.9))
    result = build_token_timestamps(trace, _tokenizer())['timestamps'][0]
    assert result['start'] == 0.48 and result['end'] == 0.64
    assert result['primary_span_index'] == 1 and result['ambiguous']
    assert [(span['start'], span['end']) for span in result['spans']] == [(0.0, 0.08), (0.48, 0.64)]


def test_reported_115k_sequence_keeps_all_56_positions_when_middle_and_final_tokens_have_no_evidence():
    # Input IDs from the reported timestamp example; the intervals below only
    # exercise index mapping, not independently verified acoustic boundaries.
    expected = [
        1073,
        8056,
        27231,
        1278,
        31134,
        1044,
        1321,
        1278,
        24349,
        1584,
        3746,
        24489,
        26236,
        1044,
        1809,
        1278,
        6138,
        1513,
        1032,
        1057,
        1050,
        1049,
        1046,
        1054,
        1053,
        1055,
        4372,
        1070,
        1395,
        12593,
        4455,
        1278,
        13472,
        1046,
        9246,
        1639,
        4206,
        1278,
        12450,
        1513,
        1032,
        1049,
        1057,
        1050,
        1046,
        1048,
        1046,
        1050,
        1046,
        1050,
        1050,
        1052,
        1394,
        2081,
        9576,
        1046,
    ]
    missing = {6, 9, 17, 53, 55}
    tokens = [
        _token(index, token_id, [] if index in missing else [_span(index * 0.08, (index + 1) * 0.08)])
        for index, token_id in enumerate(expected)
    ]
    for index in missing:
        tokens[index]['reason'] = 'never_selected'
    trace = dict(audio_duration_seconds=56 * 0.08, frame_step_seconds=0.08, timestamps=tokens)
    result = build_token_timestamps(trace, expected_input_token_ids=expected)['timestamps']
    assert len(result) == 56
    assert [token['token_id'] for token in result] == expected
    assert [token['token_index'] for token in result] == list(range(56))
    assert {token['token_index'] for token in result if token['start'] is None} == missing
    assert all(result[index]['reason'] == 'never_selected' for index in missing)
    assert result[54]['start'] == pytest.approx(54 * 0.08)
    assert result[18]['token_id'] == result[40]['token_id'] == 1032  # Distinct spacing-token occurrences.
    assert result[18]['start'] != result[40]['start']


def test_multiturn_crop_rebases_spans_and_clips_partial_final_frame():
    trace = _trace()
    trace['audio_duration_seconds'] = 1.2
    trace['timestamps'].extend(
        [
            _token(0, 1, [_span(0.88, 0.96)], turn=1),
            _token(1, 2, [_span(1.04, 1.2)], turn=1),
            _token(2, 3, [], turn=1),
        ]
    )
    result = build_token_timestamps(
        trace, _tokenizer(), start_time=0.9, end_time=1.15, turn_index=1, expected_input_token_ids=[1, 2, 3, 1]
    )
    assert result['audio_duration_seconds'] == pytest.approx(0.25)
    assert {token['turn_index'] for token in result['timestamps']} == {1}
    first, last, punctuation, tail = result['timestamps']
    assert (first['start'], first['end']) == pytest.approx((0.0, 0.06))
    assert (last['start'], last['end']) == pytest.approx((0.14, 0.25))
    assert first['spans'][0]['clipped'] and last['spans'][0]['clipped']
    assert punctuation['start'] is None and tail['status'] == 'unobserved'
    assert result['audio_offset_seconds'] == 0.9 and result['source_audio_duration_seconds'] == 1.2


def test_zero_audio_keeps_requested_tokens_without_fabricated_boundaries():
    trace = _trace()
    trace['audio_duration_seconds'] = 0.0
    result = build_token_timestamps(trace, _tokenizer(), expected_input_token_ids=[1, 2, 1, 3])
    assert result['audio_duration_seconds'] == 0
    assert len(result['timestamps']) == 4
    assert all(token['start'] is token['end'] is None and not token['spans'] for token in result['timestamps'])


def test_weakly_supported_monotonic_skip_keeps_null_times_and_its_evidence_reason():
    trace = _trace()
    trace.update(method='monotonic', force_progress=True, skip_penalty=6.0)
    trace['timestamps'][3].update(reason='never_selected', posterior_support=0.001, observed_frames=8)
    result = build_token_timestamps(trace, expected_input_token_ids=[1, 2, 1, 3])
    token = result['timestamps'][3]
    assert token['token_index'] == 3 and token['start'] is token['end'] is None
    assert token['reason'] == 'never_selected' and token['posterior_support'] == 0.001
    assert result['method'] == 'monotonic' and result['force_progress']


def test_forced_timestamp_keeps_raw_attention_probability_and_exports_zero_confidence():
    trace = _trace()
    span = trace['timestamps'][0]['spans'][0]
    span.update(forced=True, confidence=0.12, attention_confidence=0.12)
    before = copy.deepcopy(trace)
    token = build_token_timestamps(trace)['timestamps'][0]
    assert token['forced'] and token['confidence'] == 0.0
    assert token['attention_confidence'] == 0.12
    assert token['spans'][0]['forced'] and token['spans'][0]['confidence'] == 0.0
    assert token['spans'][0]['attention_confidence'] == 0.12
    assert trace == before


def test_mixed_forced_spans_zero_whole_token_confidence_and_crop_recomputes_flags():
    trace = _trace()
    token = trace['timestamps'][0]
    token.update(forced=True, confidence=0.0, attention_confidence=0.8)
    token['spans'][0].update(forced=False, confidence=0.8, attention_confidence=0.8)
    token['spans'].append(dict(_span(0.4, 0.48, confidence=0.0), forced=True, attention_confidence=0.2))
    mixed = build_token_timestamps(trace)['timestamps'][0]
    assert mixed['forced'] and mixed['confidence'] == 0.0
    assert mixed['primary_span_index'] == 0 and mixed['attention_confidence'] == 0.8
    supported = build_token_timestamps(trace, end_time=0.08)['timestamps'][0]
    assert not supported['forced'] and supported['confidence'] == 0.8
    forced = build_token_timestamps(trace, start_time=0.42, end_time=0.46)['timestamps'][0]
    assert forced['forced'] and forced['confidence'] == 0.0
    assert forced['attention_confidence'] == 0.2 and forced['spans'][0]['clipped']
    assert (forced['start'], forced['end']) == pytest.approx((0.0, 0.04))
    absent = build_token_timestamps(trace, start_time=0.5, end_time=0.6)['timestamps'][0]
    assert absent['start'] is absent['end'] is absent['confidence'] is absent['attention_confidence'] is None
    assert not absent['forced']


def test_legacy_token_forced_flag_is_preserved_without_span_metadata():
    trace = _trace()
    token = trace['timestamps'][0]
    token.update(start=0.0, end=0.08, spans=[], forced=True, confidence=0.3)
    result = build_token_timestamps(trace)['timestamps'][0]
    assert result['forced'] and result['confidence'] == 0.0
    assert result['attention_confidence'] == 0.3
    assert result['spans'][0]['forced'] and result['spans'][0]['confidence'] == 0.0


def test_forced_zero_confidence_does_not_change_primary_span_evidence_ranking():
    trace = _trace()
    token = trace['timestamps'][0]
    token['spans'][0].update(confidence=0.1, forced=False, attention_confidence=0.1)
    token['spans'].append(dict(_span(0.4, 0.56, confidence=0.0), forced=True, attention_confidence=0.3))
    result = build_token_timestamps(trace)['timestamps'][0]
    assert result['primary_span_index'] == 1
    assert (result['start'], result['end']) == pytest.approx((0.4, 0.56))
    assert result['forced'] and result['confidence'] == 0.0 and result['attention_confidence'] == 0.3


def test_unforced_and_unobserved_records_keep_distinct_confidence_semantics():
    result = build_token_timestamps(_trace(), expected_input_token_ids=[1, 2, 1, 3, 1])['timestamps']
    supported = result[0]
    assert not supported['forced'] and supported['confidence'] == supported['attention_confidence'] == 0.5
    assert not supported['spans'][0]['forced'] and supported['spans'][0]['attention_confidence'] == 0.5
    for token in result[3:]:
        assert token['start'] is token['end'] is token['confidence'] is token['attention_confidence'] is None
        assert not token['forced']


def test_crop_updates_primary_short_evidence_and_does_not_keep_ambiguity_without_visible_spans():
    trace = _trace()
    token = trace['timestamps'][0]
    token.update(short_evidence=False, ambiguous=True)
    token['spans'][0]['short_evidence'] = False
    token['spans'].append(dict(_span(0.4, 0.48, confidence=0.2), short_evidence=True))
    cropped = build_token_timestamps(trace, start_time=0.4, end_time=0.48)['timestamps'][0]
    assert cropped['short_evidence'] and not cropped['ambiguous']
    assert (cropped['start'], cropped['end']) == pytest.approx((0.0, 0.08))
    absent = build_token_timestamps(trace, start_time=0.5, end_time=0.6)['timestamps'][0]
    assert absent['start'] is None and not absent['ambiguous'] and not absent['short_evidence']


def test_mismatched_input_id_or_index_is_rejected_instead_of_attaching_shifted_times():
    with pytest.raises(ValueError, match='does not match'):
        build_token_timestamps(_trace(), expected_input_token_ids=[2, 1, 1, 3])
    with pytest.raises(ValueError, match='beyond the requested'):
        build_token_timestamps(_trace(), expected_input_token_ids=[1, 2])


def test_sidecar_is_absolute_json_and_rerun_replaces_it_atomically(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    audio = tmp_path / 'outputs' / 'predicted_audio_0.wav'
    audio.parent.mkdir()
    audio.write_bytes(b'wav')
    path = save_token_timestamps('outputs/predicted_audio_0.wav', _trace(), tokenizer=_tokenizer())
    assert path == str(audio.with_suffix('.timestamps.json'))
    record = json.loads(audio.with_suffix('.timestamps.json').read_text())
    assert record['predicted_audio_filepath'] == str(audio)
    assert record['source'] == 'attention' and record['timings_are_estimates']
    assert 'timestamps' in record and 'alignment' not in record
    save_token_timestamps(audio, _trace(), tokenizer=_tokenizer(), end_time=0.08)
    second = json.loads(audio.with_suffix('.timestamps.json').read_text())
    assert second['timestamps'][1]['start'] is None
    assert not list(audio.parent.glob('*.tmp'))


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_invalid_crop_times_fail_without_publishing_json(value):
    with pytest.raises(ValueError, match='must be finite'):
        build_token_timestamps(_trace(), start_time=value)


def test_missing_timestamp_capture_is_not_reported_as_empty_success():
    with pytest.raises(ValueError, match='returned no timestamps'):
        build_token_timestamps(None)


def test_frame_history_exports_instantaneous_confidence_and_null_labels_without_mutating_trace():
    trace = _trace_with_frames()
    before = copy.deepcopy(trace)
    result = build_token_timestamps(trace, _tokenizer())
    first, missing, forced, partial = result['frames']
    assert first['token'] == 'a' and first['confidence'] == 0.7
    assert first['confidence'] != result['timestamps'][0]['confidence']
    assert missing['status'] == 'unaligned' and missing['token_id'] is missing['confidence'] is None
    assert forced['forced'] and forced['confidence'] == 0.0 and forced['attention_confidence'] == 0.25
    assert forced['assignments'][0]['confidence'] == 0.0
    assert partial['valid_codec_frames'] == 1 and partial['end'] == pytest.approx(0.28)
    assert result['frame_count'] == result['source_frame_count'] == 4
    assert result['frame_confidence_kind'] == 'instantaneous_full_key_attention_probability'
    assert trace == before


def test_frame_crop_filters_turns_and_rebases_codec_grid_indices_and_availability():
    trace = _trace_with_frames()
    trace['turns'].append(dict(turn_index=1, source_token_ids=[2]))
    next_turn_frame = dict(_frame(4, 0.32, 0.4, 1, 2, 0.8), turn_index=1, turn_frame_index=0)
    next_turn_frame['assignments'] = [copy.deepcopy(next_turn_frame)]
    trace['frames'].append(next_turn_frame)
    trace['frame_count'] = 5
    result = build_token_timestamps(trace, start_time=0.32, end_time=0.4, turn_index=1)
    assert len(result['frames']) == result['frame_count'] == 1
    frame = result['frames'][0]
    assert frame['turn_index'] == 1 and frame['frame_index'] == 0 and frame['source_frame_index'] == 4
    assert (frame['start'], frame['end'], frame['emitted_at']) == pytest.approx((0.0, 0.08, 0.08))
    assert (frame['codec_frame_start'], frame['codec_frame_end'], frame['valid_codec_frames']) == (0, 2, 2)
    assert (frame['source_codec_frame_start'], frame['source_codec_frame_end']) == (8, 10)
    assert frame['source_valid_codec_frames'] == 2 and result['source_frame_count'] == 5
    assert (frame['assignments'][0]['start'], frame['assignments'][0]['end']) == pytest.approx((0.0, 0.08))


def test_arbitrary_frame_crop_keeps_exact_seconds_and_original_emission_time():
    trace = _trace_with_frames()
    result = build_token_timestamps(trace, start_time=0.025, end_time=0.06)
    frame = result['frames'][0]
    assert len(result['frames']) == 1 and frame['clipped']
    assert (frame['start'], frame['end']) == pytest.approx((0.0, 0.035))
    assert frame['emitted_at'] == pytest.approx(0.055)  # Available after the saved audio endpoint.
    assert frame['codec_frame_start'] is frame['codec_frame_end'] is frame['valid_codec_frames'] is None
    assert (frame['source_codec_frame_start'], frame['source_codec_frame_end']) == (0, 2)
    assert frame['assignments'][0]['emitted_at'] == pytest.approx(0.055)


def test_partial_forced_frame_keeps_only_retained_subassignments_and_their_endpoint():
    trace = _trace_with_frames()
    forced = trace['frames'][2]
    forced['text_key_position'] = 12
    forced['assignments'] = [
        dict(_frame(2, 0.16, 0.20, 1, 2, 0.12), forced=True, emitted_at=0.24, text_key_position=11),
        dict(_frame(2, 0.20, 0.24, 2, 1, 0.25), forced=True, emitted_at=0.24, text_key_position=12),
    ]
    result = build_token_timestamps(trace, start_time=0.16, end_time=0.20)
    frame = result['frames'][0]
    assert len(frame['assignments']) == 1 and frame['token_index'] == 1 and frame['token_id'] == 2
    assert frame['text_key_position'] == frame['assignments'][0]['text_key_position'] == 11
    assert frame['confidence'] == 0.0 and frame['attention_confidence'] == 0.12 and frame['forced']
    assert frame['emitted_at'] == pytest.approx(0.08)
    assert frame['valid_codec_frames'] == 1 and frame['source_valid_codec_frames'] == 2


def test_drained_only_frame_history_preserves_total_count_and_does_not_invent_crop_count():
    trace = _trace_with_frames()
    trace.update(frames=[], frame_history_retained=False)
    whole = build_token_timestamps(trace)
    assert whole['frames'] == [] and whole['frame_count'] == whole['source_frame_count'] == 4
    assert not whole['frame_history_retained']
    cropped = build_token_timestamps(trace, start_time=0.08, end_time=0.16)
    assert cropped['frames'] == [] and cropped['frame_count'] is None and cropped['source_frame_count'] == 4


def _frame(index, start, end, token_index, token_id, confidence):
    return dict(
        frame_index=index,
        turn_frame_index=index,
        batch_index=0,
        turn_index=0,
        start=start,
        end=end,
        emitted_at=end,
        codec_frame_start=round(start / 0.04),
        codec_frame_end=round(end / 0.04),
        valid_codec_frames=round((end - start) / 0.04),
        token_index=token_index,
        token_id=token_id,
        token=None,
        confidence=confidence,
        attention_confidence=confidence,
        forced=False,
        text_attention_mass=0.9,
        margin=0.2 if confidence is not None else None,
        status='aligned' if token_id is not None else 'unaligned',
    )


def _trace_with_frames():
    trace = _trace()
    frames = [
        _frame(0, 0.0, 0.08, 0, 1, 0.7),
        _frame(1, 0.08, 0.16, None, None, None),
        dict(_frame(2, 0.16, 0.24, 2, 1, 0.25), forced=True),
        _frame(3, 0.24, 0.28, 2, 1, 0.4),
    ]
    for frame in frames:
        frame['assignments'] = [copy.deepcopy(frame)]
    trace.update(frames=frames, frame_count=4, frame_history_retained=True, codec_frame_seconds=0.04)
    return trace


def _tokenizer():
    return SimpleNamespace(tokens=['<pad>', 'a', 'b', '.'])


def _span(start, end, confidence=0.5):
    return dict(start=start, end=end, confidence=confidence, text_attention_mass=0.9, frame_count=2, confirmed=True)


def _token(index, token_id, spans, turn=0):
    return dict(
        source='attention',
        turn_index=turn,
        token_index=index,
        token_id=token_id,
        start=spans[0]['start'] if spans else None,
        end=spans[0]['end'] if spans else None,
        confidence=0.5 if spans else None,
        status='finalized' if spans else 'unaligned',
        spans=spans,
    )


def _trace():
    return dict(
        source='attention',
        audio_duration_seconds=0.64,
        frame_step_seconds=0.08,
        attention_layer='middle',
        attention_head=0,
        confirmation_frames=2,
        turns=[dict(turn_index=0, source_token_ids=[1, 2, 1, 3])],
        timestamps=[
            _token(0, 1, [_span(0.0, 0.08)]),
            _token(1, 2, [_span(0.08, 0.24)]),
            _token(2, 1, [_span(0.24, 0.32)]),
            _token(3, 3, []),
        ],
    )
