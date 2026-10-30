# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serialize generated token traces as audio-relative debugging estimates."""

import copy
import json
import math
from pathlib import Path


def build_token_alignment_debug(debug, model, start_time=0.0, end_time=None):
    """Clip a model trace to one saved waveform and construct token spans.

    Head tokens are output runs, not forced-aligned input positions. Attention
    tokens retain input positions, including separate occurrences of the same ID.
    Times are estimates on the codec grid and never verified word boundaries.
    """
    if debug is None:
        raise ValueError('Token alignment recording was requested but the model returned no alignment_debug trace')
    duration = float(debug['audio_duration_seconds'])
    start_time = max(0.0, min(float(start_time), duration))
    end_time = duration if end_time is None else max(start_time, min(float(end_time), duration))
    result = copy.deepcopy(debug)
    codec_step = float(debug['frame_step_seconds']) / int(debug.get('frame_stacking_factor', 1))
    result['source_audio_duration_seconds'] = duration
    if 'audio_frame_count' in debug:
        result['source_audio_frame_count'] = debug['audio_frame_count']
    result['audio_duration_seconds'] = end_time - start_time
    result['audio_frame_count'] = max(0, math.ceil((end_time - start_time) / codec_step - 1e-8))
    result['codec_frame_step_seconds'] = codec_step
    result['frame_coordinate_system'] = 'codec bins relative to the saved waveform; seconds are authoritative'
    result['audio_offset_seconds'] = start_time
    result['timing_kind'] = 'model_estimate'
    result['inference_settings'] = dict(getattr(model, 'attention_alignment_inference_config', {}))
    frames = []
    for raw_frame in result.get('frames', []):
        left, right = max(float(raw_frame['start']), start_time), min(float(raw_frame['end']), end_time)
        if right <= left:
            continue
        frame = dict(raw_frame, start=left - start_time, end=right - start_time)
        for name in ('start_frame', 'end_frame', 'decoder_step'):
            if name in frame:
                frame['source_' + name] = frame[name]
        frame['start_frame'] = max(0, math.floor(frame['start'] / codec_step + 1e-8))
        frame['end_frame'] = min(result['audio_frame_count'], math.ceil(frame['end'] / codec_step - 1e-8))
        frame['decoder_step'] = max(0, math.floor(frame['start'] / float(debug['frame_step_seconds']) + 1e-8))
        head = frame.get('text_head')
        if head is not None:
            frame['text_head'] = dict(head, token=_token_label(model.tokenizer, head['token_id']))
        frames.append(frame)
    result['frames'] = frames
    active_turns = {frame.get('turn_index', 0) for frame in frames}
    result['turns'] = [turn for turn in result.get('turns', []) if turn['turn_index'] in active_turns]

    head_runs = _runs(frames, 'text_head', 'token_id')
    ignored = {getattr(model, name, None) for name in ('pad_id', 'bos_id', 'eos_id', 'cfg_unk_token_id')}
    for run in head_runs:
        run['token'] = _token_label(model.tokenizer, run['token_id'])
        run['is_special'] = run['token_id'] in ignored
    content_ids = [run['token_id'] for run in head_runs if not run['is_special']]
    result['text_head'] = {
        'decoded_text': _decode_ids(model.tokenizer, content_ids),
        'token_ids': content_ids,
        'tokens': head_runs,
        'position_kind': 'consecutive_output_runs',
        'note': 'Adjacent identical input tokens cannot be separated from repeated head IDs alone.',
    }

    segments = _runs(frames, 'attention', 'attention_token_index')
    source_turns = {turn['turn_index']: turn for turn in result.get('turns', [])}
    tokens = []
    for turn_index, turn in source_turns.items():
        for index, token_id in enumerate(turn.get('source_token_ids', [])):
            matching = [s for s in segments if s['turn_index'] == turn_index and s['attention_token_index'] == index]
            spans = [{'start': segment['start'], 'end': segment['end']} for segment in matching]
            tokens.append(
                {
                    'turn_index': turn_index,
                    'token_index': index,
                    'token_id': token_id,
                    'token': _token_label(model.tokenizer, token_id),
                    'start': spans[0]['start'] if spans else None,
                    'end': spans[-1]['end'] if spans else None,
                    'spans': spans,
                }
            )
    result['attention'] = {
        'tokens': tokens,
        'segments': segments,
        'position_kind': 'input_bpe_position_before_window',
        'note': 'Argmax attention estimates; disjoint spans preserve backward movements and revisits.',
    }
    return result


def save_token_alignment_debug(
    audio_path,
    debug,
    model,
    start_time=0.0,
    end_time=None,
    expected_input_text=None,
    expected_input_token_ids=None,
):
    """Write a JSON sidecar atomically next to a generated waveform."""
    audio_path = Path(audio_path).resolve()
    path = audio_path.with_suffix('.alignment.json')
    record = build_token_alignment_debug(debug, model, start_time=start_time, end_time=end_time)
    record['predicted_audio_filepath'] = str(audio_path)
    record['source_token_scope'] = 'observed_streaming_input'
    if expected_input_text is not None:
        record['expected_input_text'] = expected_input_text
    if expected_input_token_ids is not None:
        record['expected_input_token_ids'] = list(expected_input_token_ids)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)
    return str(path)


def _runs(frames, channel, identity_key):
    runs = []
    for frame in frames:
        value = frame.get(channel)
        if value is None or value.get(identity_key) is None:
            continue
        identity = value[identity_key]
        if identity < 0:
            continue
        turn = frame.get('turn_index', 0)
        if (
            runs
            and runs[-1][identity_key] == identity
            and runs[-1]['turn_index'] == turn
            and abs(runs[-1]['end'] - frame['start']) < 1e-8
        ):
            runs[-1]['end'] = frame['end']
            runs[-1]['frame_count'] += 1
        else:
            runs.append(
                {
                    'turn_index': turn,
                    identity_key: identity,
                    'start': frame['start'],
                    'end': frame['end'],
                    'frame_count': 1,
                }
            )
    return runs


def _token_label(tokenizer, token_id):
    component, local_id, _ = _tokenizer_component(tokenizer, token_id)
    convert = getattr(component, 'convert_ids_to_tokens', None)
    if callable(convert):
        return str(convert(local_id))
    tokens = getattr(component, 'tokens', None)
    if tokens is not None and 0 <= local_id < len(tokens):
        return str(tokens[local_id])
    return str(token_id)


def _decode_ids(tokenizer, token_ids):
    groups = []
    for token_id in token_ids:
        component, local_id, name = _tokenizer_component(tokenizer, token_id)
        if groups and groups[-1][0] == name:
            groups[-1][2].append(local_id)
        else:
            groups.append((name, component, [local_id]))
    pieces = []
    for _, component, ids in groups:
        decode = getattr(component, 'decode', None) or getattr(component, 'ids_to_text', None)
        pieces.append(
            decode(ids) if callable(decode) else ''.join(_token_label(component, token_id) for token_id in ids)
        )
    return ''.join(pieces)


def _tokenizer_component(tokenizer, token_id):
    components = getattr(tokenizer, 'tokenizers', None)
    if components is not None:
        for name, component in components.items():
            offset = tokenizer.tokenizer_offsets[name]
            if offset <= token_id < offset + tokenizer.num_tokens_per_tokenizer[name]:
                return component, token_id - offset, name
        return None, token_id, None
    return tokenizer, token_id, None
