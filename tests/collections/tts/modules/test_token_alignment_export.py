# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from types import SimpleNamespace

import pytest

from nemo.collections.tts.modules.magpietts_inference.token_alignment import (
    build_token_alignment_debug,
    save_token_alignment_debug,
)

pytestmark = pytest.mark.unit


def test_head_runs_and_attention_positions_preserve_different_repetition_semantics():
    debug = _trace()
    before = copy.deepcopy(debug)
    result = build_token_alignment_debug(debug, _model())
    assert result['text_head']['token_ids'] == [1, 2, 1]
    assert result['text_head']['decoded_text'] == 'aba'
    assert [(token['token_id'], token['start'], token['end']) for token in result['text_head']['tokens']] == [
        (1, 0.0, 0.16),
        (2, 0.16, 0.24),
        (1, 0.24, 0.3),
    ]
    attention = result['attention']['tokens']
    assert [token['token_id'] for token in attention] == [1, 1, 2]
    assert [token['token_index'] for token in attention] == [0, 1, 2]
    assert attention[0]['spans'] == [{'start': 0.0, 'end': 0.08}, {'start': 0.24, 'end': 0.3}]
    assert attention[1]['spans'] == [{'start': 0.08, 'end': 0.16}]
    assert debug == before


def test_export_clips_partial_last_frame_and_rebases_multiturn_waveform():
    debug = _trace()
    debug['turns'].append(dict(turn_index=1, source_token_ids=[2], source_key_positions=[50]))
    debug['frames'][-1]['turn_index'] = 1
    debug['frames'][-1]['attention']['attention_token_index'] = 0
    result = build_token_alignment_debug(debug, _model(), start_time=0.25, end_time=0.29)
    assert result['audio_duration_seconds'] == pytest.approx(0.04)
    assert len(result['frames']) == 1
    assert result['frames'][0]['start'] == 0.0
    assert result['frames'][0]['end'] == pytest.approx(0.04)
    assert [turn['turn_index'] for turn in result['turns']] == [1]
    assert result['attention']['tokens'][0]['turn_index'] == 1


def test_multiturn_codec_indices_are_rebased_and_partial_stack_end_is_clipped():
    debug = _trace()
    debug.update(frame_stacking_factor=2, audio_frame_count=8)
    for index, frame in enumerate(debug['frames']):
        frame.update(start_frame=index * 2, end_frame=(index + 1) * 2, decoder_step=100 + index)
    result = build_token_alignment_debug(debug, _model(), start_time=0.08, end_time=0.28)
    assert result['audio_frame_count'] == 5
    assert result['source_audio_frame_count'] == 8
    assert result['frames'][0]['start_frame'] == 0
    assert result['frames'][0]['source_start_frame'] == 2
    assert result['frames'][0]['decoder_step'] == 0
    assert result['frames'][0]['source_decoder_step'] == 101
    assert result['frames'][-1]['end_frame'] == 5
    assert result['frames'][-1]['source_end_frame'] == 8
    assert result['frames'][-1]['end'] == pytest.approx(0.2)


def test_padding_head_predictions_do_not_join_separate_runs_or_fake_input_positions():
    debug = _trace()
    debug['frames'][1]['text_head']['token_id'] = 0
    result = build_token_alignment_debug(debug, _model())
    assert result['text_head']['tokens'][1]['is_special']
    assert all('token_index' not in token for token in result['text_head']['tokens'])
    assert result['text_head']['position_kind'] == 'consecutive_output_runs'


def test_numeric_token_ids_are_decoded_with_aggregate_offsets_not_vocab_insertion_order():
    component = SimpleNamespace(
        convert_ids_to_tokens=lambda token_id: {0: 'zero', 1: 'a', 2: 'b'}[token_id],
        decode=lambda ids: ''.join({0: 'zero', 1: 'a', 2: 'b'}[token_id] for token_id in ids),
    )
    tokenizer = SimpleNamespace(
        tokenizers={'bpe': component},
        tokenizer_offsets={'bpe': 5},
        num_tokens_per_tokenizer={'bpe': 3},
        tokens=['wrong'] * 8,
    )
    model = _model()
    model.tokenizer = tokenizer
    debug = _trace()
    debug['turns'][0]['source_token_ids'] = [6, 6, 7]
    for frame in debug['frames']:
        frame['text_head']['token_id'] += 5
    result = build_token_alignment_debug(debug, model)
    assert result['text_head']['decoded_text'] == 'aba'
    assert result['text_head']['tokens'][0]['token'] == 'a'
    assert result['attention']['tokens'][2]['token'] == 'b'


def test_sidecar_is_json_and_rerun_replaces_previous_content(tmp_path):
    audio = tmp_path / 'predicted_audio_7.wav'
    path = save_token_alignment_debug(audio, _trace(), _model())
    first = json.loads(open(path, encoding='utf-8').read())
    assert first['predicted_audio_filepath'] == str(audio)
    save_token_alignment_debug(audio, _trace(), _model(), end_time=0.08)
    second = json.loads(open(path, encoding='utf-8').read())
    assert len(second['frames']) == 1
    assert not list(tmp_path.glob('*.tmp'))


def test_missing_recording_is_reported_instead_of_saving_empty_success():
    with pytest.raises(ValueError, match='no alignment_debug'):
        build_token_alignment_debug(None, _model())


def test_requested_input_metadata_distinguishes_an_unobserved_tail_and_preserves_duplicates(tmp_path):
    debug = _trace()
    debug['turns'][0]['source_token_ids'] = [1]
    expected = [1, 1, 2, 2]
    path = save_token_alignment_debug(
        tmp_path / 'predicted_audio_0.wav',
        debug,
        _model(),
        expected_input_text='aabb',
        expected_input_token_ids=expected,
    )
    record = json.loads(open(path, encoding='utf-8').read())
    assert record['source_token_scope'] == 'observed_streaming_input'
    assert record['turns'][0]['source_token_ids'] == [1]
    assert record['expected_input_text'] == 'aabb'
    assert record['expected_input_token_ids'] == expected
    assert expected == [1, 1, 2, 2]


def _model():
    return SimpleNamespace(
        tokenizer=SimpleNamespace(tokens=['<pad>', 'a', 'b'], decode=lambda ids: ''.join(' ab'[i] for i in ids)),
        pad_id=0,
        bos_id=3,
        eos_id=4,
        attention_alignment_inference_config={'enabled': False, 'record_alignments': True},
    )


def _trace():
    return {
        'schema_version': 1,
        'audio_duration_seconds': 0.3,
        'frame_step_seconds': 0.08,
        'turns': [dict(turn_index=0, source_token_ids=[1, 1, 2], source_key_positions=[10, 11, 12])],
        'frames': [
            dict(
                start=start,
                end=end,
                turn_index=0,
                text_head=dict(token_id=head, probability=0.5),
                attention=dict(attention_token_index=position, attention_probability=0.6),
            )
            for start, end, head, position in [
                (0.0, 0.08, 1, 0),
                (0.08, 0.16, 1, 1),
                (0.16, 0.24, 2, 2),
                (0.24, 0.3, 1, 0),
            ]
        ],
    }
