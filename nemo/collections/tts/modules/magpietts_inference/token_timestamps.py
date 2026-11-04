# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persist streaming attention timestamps beside an audio waveform.

Only JSON data and an optional tokenizer are required; no TTS model is loaded.
The reusable streaming estimator lives in ``tts.modules.attention_timestamps``.
"""

import copy
import json
import math
from pathlib import Path


def build_token_timestamps(
    trace,
    tokenizer=None,
    start_time=0.0,
    end_time=None,
    turn_index=None,
    expected_input_text=None,
    expected_input_token_ids=None,
):
    """Clip timestamp records to a saved waveform without manufacturing boundaries.

    ``trace`` is one batch item's timestamp output. Times are relative to its
    returned waveform. A multiturn caller selects a turn and crops that waveform;
    all saved times are then rebased to the cropped audio. Requested but unseen
    input occurrences remain present with null times and ``status=unobserved``.
    Heuristically forced spans have zero ``confidence``; ``attention_confidence``
    retains their underlying attention probability for debugging. Any retained
    forced span makes the containing token forced, even if its primary span has
    direct attention support. Frame records preserve each causal decoder query,
    including its instantaneous score and availability time. These scores are
    not calibrated correctness odds.
    """
    if trace is None:
        raise ValueError('Timestamp recording was requested but the model returned no timestamps')
    duration = float(trace['audio_duration_seconds'])
    if not math.isfinite(duration) or duration < 0:
        raise ValueError('Timestamp audio duration must be finite and nonnegative')
    start_time = float(start_time)
    end_time = duration if end_time is None else float(end_time)
    if not math.isfinite(start_time) or not math.isfinite(end_time):
        raise ValueError('Timestamp crop times must be finite')
    start_time = max(0.0, min(start_time, duration))
    end_time = max(start_time, min(end_time, duration))
    result = {
        key: copy.deepcopy(value)
        for key, value in trace.items()
        if key not in ('timestamps', 'turns', 'events', 'frames')
    }
    result.update(
        schema_version=1,
        source='attention',
        timings_are_estimates=True,
        timing_kind='attention_estimate',
        audio_duration_seconds=end_time - start_time,
        audio_offset_seconds=start_time,
        source_audio_duration_seconds=duration,
        primary_span_rule='largest probability-duration support; all disjoint spans are retained',
        confidence_kind='mean_full_key_attention_probability',
        confidence_is_calibrated=False,
    )
    if turn_index is not None:
        result['source_turn_index'] = turn_index
    turns = [
        copy.deepcopy(turn)
        for turn in trace.get('turns', [])
        if turn_index is None or turn.get('turn_index', 0) == turn_index
    ]
    result['turns'] = turns
    records = [
        copy.deepcopy(token)
        for token in trace.get('timestamps', [])
        if turn_index is None or token.get('turn_index', 0) == turn_index
    ]
    timestamps = []
    for token in records:
        token.setdefault('turn_index', 0 if turn_index is None else turn_index)
        token.setdefault('observed', True)
        source_spans = token['spans']
        spans = []
        for span in source_spans:
            clipped = _crop_interval(span, start_time, end_time)
            if clipped is None:
                continue
            if clipped.get('clipped'):
                frame_step = float(trace.get('frame_step_seconds', 0.0))
                if 'frame_count' in clipped and frame_step > 0:
                    clipped['source_frame_count'] = clipped['frame_count']
                    clipped['frame_count'] = min(
                        clipped['frame_count'],
                        max(1, math.ceil((clipped['end'] - clipped['start']) / frame_step - 1e-8)),
                    )
            spans.append(clipped)
        token['spans'] = spans
        token['source'] = 'attention'
        if spans:
            primary = max(
                range(len(spans)),
                key=lambda index: (
                    float(spans[index].get('attention_confidence') or 0.0)
                    * (spans[index]['end'] - spans[index]['start']),
                    -spans[index]['start'],
                ),
            )
            forced = any(span['forced'] for span in spans)
            token.update(
                start=spans[primary]['start'],
                end=spans[primary]['end'],
                confidence=0.0 if forced else spans[primary].get('confidence'),
                attention_confidence=spans[primary].get('attention_confidence'),
                forced=forced,
                text_attention_mass=spans[primary].get('text_attention_mass', token.get('text_attention_mass')),
                primary_span_index=primary,
                ambiguous=len(spans) > 1,
                short_evidence=spans[primary].get('short_evidence', token.get('short_evidence', False)),
            )
        else:
            token.update(
                start=None,
                end=None,
                confidence=None,
                attention_confidence=None,
                forced=False,
                text_attention_mass=None,
                primary_span_index=None,
                ambiguous=False,
                short_evidence=False,
            )
            if source_spans:
                token['status'] = 'outside_audio_slice'
                token['reason'] = 'outside_audio_slice'
        token['token'] = _label(tokenizer, token['token_id'], token.get('token'))
        timestamps.append(token)
    if expected_input_token_ids is not None:
        expected = list(expected_input_token_ids)
        indices = {(token.get('turn_index', 0), token['token_index']): token for token in timestamps}
        selected_turns = {token.get('turn_index', 0) for token in timestamps}
        if turn_index is None and len(selected_turns) > 1:
            raise ValueError('Expected input IDs require selecting one turn for a multiturn timestamp trace')
        selected = turn_index if turn_index is not None else next(iter(selected_turns), 0)
        if any(token['token_index'] >= len(expected) for token in timestamps):
            raise ValueError('Timestamp trace contains positions beyond the requested input sequence')
        for index, token_id in enumerate(expected):
            token = indices.get((selected, index))
            if token is not None:
                if token['token_id'] != token_id:
                    raise ValueError(f'Timestamp token {index} does not match the requested input token')
                continue
            timestamps.append(
                {
                    'source': 'attention',
                    'turn_index': selected,
                    'token_index': index,
                    'token_id': token_id,
                    'token': _label(tokenizer, token_id),
                    'start': None,
                    'end': None,
                    'confidence': None,
                    'attention_confidence': None,
                    'forced': False,
                    'text_attention_mass': None,
                    'status': 'unobserved',
                    'reason': 'not_observed',
                    'observed': False,
                    'ambiguous': False,
                    'short_evidence': False,
                    'primary_span_index': None,
                    'spans': [],
                }
            )
        result['expected_input_token_ids'] = expected
        result['token_scope'] = 'requested_input'
    else:
        result['token_scope'] = 'observed_streaming_input'
    result['timestamps'] = sorted(timestamps, key=lambda token: (token.get('turn_index', 0), token['token_index']))
    if 'frames' in trace:
        result['frames'] = _crop_frames(trace, tokenizer, start_time, end_time, turn_index)
        result['source_frame_count'] = trace.get('frame_count', len(trace['frames']))
        history_retained = trace.get('frame_history_retained', True)
        if history_retained:
            result['frame_count'] = len(result['frames'])
        elif turn_index is not None or start_time != 0.0 or end_time != duration:
            result['frame_count'] = None
        result['frame_confidence_kind'] = 'instantaneous_full_key_attention_probability'
    if expected_input_text is not None:
        result['expected_input_text'] = expected_input_text
    return result


def save_token_timestamps(audio_path, trace, tokenizer=None, **kwargs):
    """Atomically save ``<audio stem>.timestamps.json`` and return its absolute path."""
    audio_path = Path(audio_path).resolve()
    destination = audio_path.with_suffix('.timestamps.json')
    record = build_token_timestamps(trace, tokenizer=tokenizer, **kwargs)
    record['predicted_audio_filepath'] = str(audio_path)
    temporary = destination.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(destination)
    return str(destination)


def _crop_frames(trace, tokenizer, start_time, end_time, turn_index):
    frames = []
    codec_step = float(trace.get('codec_frame_seconds', 0.0))
    for original in trace['frames']:
        if turn_index is not None and original.get('turn_index', 0) != turn_index:
            continue
        frame = _crop_interval(original, start_time, end_time)
        if frame is None:
            continue
        frame['source_frame_index'] = original['frame_index']
        frame['frame_index'] = len(frames)
        for name in ('codec_frame_start', 'codec_frame_end', 'valid_codec_frames'):
            if name in original:
                frame[f'source_{name}'] = original[name]
        # Seconds remain exact for arbitrary waveform crops. Integer codec
        # indices describe only intervals whose boundaries lie on that grid.
        left = _codec_grid_index(frame['start'], codec_step)
        right = _codec_grid_index(frame['end'], codec_step)
        if _codec_grid_index(start_time, codec_step) is None or left is None or right is None:
            frame.update(codec_frame_start=None, codec_frame_end=None, valid_codec_frames=None)
        else:
            frame.update(codec_frame_start=left, codec_frame_end=right, valid_codec_frames=right - left)
        assignments = []
        for assignment in original.get('assignments', []):
            clipped = _crop_interval(assignment, start_time, end_time)
            if clipped is not None:
                _label_frame(clipped, tokenizer)
                assignments.append(clipped)
        if 'assignments' in original:
            frame['assignments'] = assignments
        if assignments:
            # A partial crop may remove the source query's final subdivision.
            # Its retained endpoint is then the last surviving assignment.
            for name in (
                'token_index',
                'token_id',
                'token',
                'text_key_position',
                'confidence',
                'attention_confidence',
                'forced',
                'text_attention_mass',
                'margin',
                'status',
            ):
                if name in assignments[-1]:
                    frame[name] = assignments[-1][name]
        _label_frame(frame, tokenizer)
        frames.append(frame)
    return frames


def _crop_interval(record, start_time, end_time):
    left, right = max(float(record['start']), start_time), min(float(record['end']), end_time)
    if right <= left:
        return None
    clipped = copy.deepcopy(record)
    clipped.update(start=left - start_time, end=right - start_time)
    if left != record['start'] or right != record['end']:
        clipped['clipped'] = True
    if record.get('emitted_at') is not None:
        clipped['emitted_at'] = float(record['emitted_at']) - start_time
    if clipped.get('forced', False):
        clipped['confidence'] = 0.0
    return clipped


def _codec_grid_index(seconds, step):
    if not math.isfinite(step) or step <= 0:
        return None
    value = seconds / step
    rounded = round(value)
    return rounded if abs(value - rounded) <= 1e-7 else None


def _label_frame(record, tokenizer):
    if record.get('token_id') is not None:
        record['token'] = _label(tokenizer, record['token_id'], record.get('token'))


def _label(tokenizer, token_id, existing=None):
    if existing is not None:
        return existing
    return _token_label(tokenizer, token_id) if tokenizer is not None else str(token_id)


def _token_label(tokenizer, token_id):
    components = getattr(tokenizer, 'tokenizers', None)
    if components is not None:
        for name, component in components.items():
            offset = tokenizer.tokenizer_offsets[name]
            if offset <= token_id < offset + tokenizer.num_tokens_per_tokenizer[name]:
                tokenizer, token_id = component, token_id - offset
                break
        else:
            return str(token_id)
    convert = getattr(tokenizer, 'convert_ids_to_tokens', None)
    if callable(convert):
        return str(convert(token_id))
    tokens = getattr(tokenizer, 'tokens', None)
    if tokens is not None and 0 <= token_id < len(tokens):
        return str(tokens[token_id])
    return str(token_id)
