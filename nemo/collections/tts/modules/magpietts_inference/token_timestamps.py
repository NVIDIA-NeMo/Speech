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
    direct attention support. These scores are not calibrated correctness odds.
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
        key: copy.deepcopy(value) for key, value in trace.items() if key not in ('timestamps', 'turns', 'events')
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
        source_spans = token.get('spans') or []
        if not source_spans and token.get('start') is not None and token.get('end') is not None:
            source_spans = [
                dict(
                    start=token['start'],
                    end=token['end'],
                    confidence=token.get('confidence'),
                    attention_confidence=token.get('attention_confidence', token.get('confidence')),
                    forced=bool(token.get('forced', False)),
                )
            ]
        # Older callers may have marked only the token. Once span-level flags
        # exist they are authoritative, so cropping away a forced span also
        # clears the token's forced flag.
        legacy_forced = bool(token.get('forced', False)) and not any('forced' in span for span in source_spans)
        spans = []
        for span in source_spans:
            left = max(float(span['start']), start_time)
            right = min(float(span['end']), end_time)
            if right <= left:
                continue
            clipped = dict(span, start=left - start_time, end=right - start_time)
            clipped['forced'] = bool(span.get('forced', legacy_forced))
            clipped['attention_confidence'] = span.get('attention_confidence', span.get('confidence'))
            if clipped['forced']:
                clipped['confidence'] = 0.0
            if left != span['start'] or right != span['end']:
                clipped['clipped'] = True
                frame_step = float(trace.get('frame_step_seconds', 0.0))
                if 'frame_count' in clipped and frame_step > 0:
                    clipped['source_frame_count'] = clipped['frame_count']
                    clipped['frame_count'] = min(
                        clipped['frame_count'], max(1, math.ceil((right - left) / frame_step - 1e-8))
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
