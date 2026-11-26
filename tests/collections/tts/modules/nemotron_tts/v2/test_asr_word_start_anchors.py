# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from nemo.collections.tts.modules.nemotron_tts.v2.timestamp_head import TTSCtcTokenizer, map_asr_word_start_anchors

pytestmark = pytest.mark.unit


def test_first_anchor_uses_ctc_onset_and_preserves_the_public_initial_silence_interval():
    tokenizer, ids, native = _tokenizer([(0, 3), (3, 7)])
    words = [dict(word='one', start=0.0, end=0.8, ctc_start=0.32), dict(word='two', start=0.8, end=1.2, ctc_start=0.8)]
    original = deepcopy(words)
    result = map_asr_word_start_anchors(tokenizer, 'one two', torch.tensor(ids), words, 1.2, 0.08)
    assert result == dict(
        anchors=[_anchor(4, 0, 1), _anchor(10, 1, 2)],
        error=None,
        token_coverage=1.0,
        merged_word_groups=0,
        collision_count=0,
    )
    assert words == original
    native.assert_called_once_with('one two', return_offsets_mapping=True, add_special_tokens=False)


@pytest.mark.parametrize('ctc_start,expected', [(None, 3), (0.0, 0)])
def test_legacy_word_start_fallback_does_not_replace_a_real_zero_ctc_onset(ctc_start, expected):
    tokenizer, ids, _ = _tokenizer([(0, 3)])
    word = dict(word='one', start=0.24, end=0.64)
    if ctc_start is not None:
        word['ctc_start'] = ctc_start
    result = map_asr_word_start_anchors(tokenizer, 'one', ids, [word], 0.64, 0.08)
    assert result['anchors'] == [_anchor(expected, 0, 1)]


@pytest.mark.parametrize('onset,expected', [(0.04, 1), (0.12, 2), (0.20, 3), (0.28, 4)])
def test_onset_rounding_ties_go_forward_despite_float_representation(onset, expected):
    tokenizer, ids, _ = _tokenizer([(0, 3)])
    result = map_asr_word_start_anchors(
        tokenizer, 'one', ids, [dict(word='one', start=0, end=0.8, ctc_start=onset)], 0.8, 0.08
    )
    assert result['anchors'] == [_anchor(expected, 0, 1)]


def test_fractional_turn_offset_is_combined_before_rounding_to_the_backbone_query_grid():
    tokenizer, ids, _ = _tokenizer([(0, 3)])
    words = [dict(word='one', start=0, end=0.4, ctc_start=0.05)]
    result = map_asr_word_start_anchors(tokenizer, 'one', ids, words, 0.4, 0.08, audio_start_seconds=0.13)
    assert result['anchors'] == [_anchor(2, 0, 1)]
    # The teacher's 40 ms CTC subframes do not define the backbone's query count.
    finer = map_asr_word_start_anchors(tokenizer, 'one', ids, words, 0.4, 0.04, audio_start_seconds=0.13)
    assert finer['anchors'] == [_anchor(5, 0, 1)]


def test_colliding_word_starts_merge_adjacent_bpe_occurrences_into_one_anchor():
    tokenizer, ids, _ = _tokenizer([(0, 1), (1, 3), (3, 5), (5, 7)])
    starts = [0.01, 0.03, 0.045, 0.17]
    ends = [*starts[1:], 0.4]
    words = [
        dict(word=word, start=0.0 if index == 0 else start, end=end, ctc_start=start)
        for index, (word, start, end) in enumerate(zip('abcd', starts, ends))
    ]
    result = map_asr_word_start_anchors(tokenizer, 'a b c d', ids, words, 0.4, 0.08)
    assert result['anchors'] == [_anchor(0, 0, 2), _anchor(1, 2, 3), _anchor(2, 3, 4)]
    assert result['collision_count'] == 1
    assert result['merged_word_groups'] == 0 and result['token_coverage'] == 1.0


def test_bpe_piece_crossing_words_reuses_existing_interval_grouping_at_the_first_onset():
    tokenizer, ids, _ = _tokenizer([(0, 5), (5, 7), (7, 13)])
    words = [
        dict(word='one', start=0, end=0.5, ctc_start=0.2),
        dict(word='two', start=0.5, end=1.0, ctc_start=0.55),
        dict(word='three', start=1.0, end=1.5, ctc_start=1.02),
    ]
    result = map_asr_word_start_anchors(tokenizer, 'one two three', ids, words, 1.5, 0.08)
    assert result['anchors'] == [_anchor(3, 0, 2), _anchor(13, 2, 3)]
    assert result['merged_word_groups'] == 1 and result['collision_count'] == 0


def test_raw_number_and_punctuation_tokens_remain_the_original_native_word_targets():
    tokenizer, ids, native = _tokenizer([(0, 1), (1, 3), (3, 4), (4, 8)])
    words = [dict(word='$10,', start=0, end=0.6, ctc_start=0.12), dict(word='now', start=0.6, end=1.0, ctc_start=0.6)]
    result = map_asr_word_start_anchors(tokenizer, '$10, now', ids, words, 1.0, 0.08)
    assert result['anchors'] == [_anchor(2, 0, 3), _anchor(8, 3, 4)]
    assert native.call_count == 1
    assert ids == [18, 19, 20, 21]


def test_repeated_token_ids_keep_separate_word_occurrences():
    tokenizer, ids, _ = _tokenizer([(0, 3), (3, 7)], local_ids=[5, 5])
    words = [dict(word='one', start=0, end=0.2, ctc_start=0.12), dict(word='one', start=0.2, end=0.5, ctc_start=0.2)]
    result = map_asr_word_start_anchors(tokenizer, 'one one', ids, words, 0.5, 0.08)
    assert result['anchors'] == [_anchor(2, 0, 1), _anchor(3, 1, 2)]


def test_unicode_byte_pieces_with_shared_offsets_keep_the_same_character_group():
    tokenizer, ids, _ = _tokenizer([(0, 1), (0, 1), (1, 2)])
    words = [dict(word='你', start=0, end=0.16, ctc_start=0.09), dict(word='好', start=0.16, end=0.4, ctc_start=0.19)]
    result = map_asr_word_start_anchors(tokenizer, '你好', ids, words, 0.4, 0.08)
    assert result['anchors'] == [_anchor(1, 0, 2), _anchor(2, 2, 3)]


@pytest.mark.parametrize('failure', ['crop', 'teacher_text', 'teacher_onset'])
def test_incompatible_word_or_native_token_mapping_returns_no_anchors(failure):
    tokenizer, ids, _ = _tokenizer([(0, 3), (3, 7)])
    words = [dict(word='one', start=0, end=0.5, ctc_start=0.1), dict(word='two', start=0.5, end=1.0, ctc_start=0.6)]
    if failure == 'crop':
        ids = ids[:-1]
    elif failure == 'teacher_text':
        words[1]['word'] = 'three'
    else:
        words[0]['ctc_start'] = float('nan')
    result = map_asr_word_start_anchors(tokenizer, 'one two', ids, words, 1.0, 0.08)
    assert (
        result['error']
        == dict(crop='input_token_mismatch', teacher_text='transcript_mismatch', teacher_onset='invalid_word_times')[
            failure
        ]
    )
    assert result['anchors'] == [] and result['token_coverage'] == 0


@pytest.mark.parametrize(
    'field,value',
    [
        ('frame_seconds', 0),
        ('frame_seconds', -0.08),
        ('frame_seconds', float('nan')),
        ('frame_seconds', float('inf')),
        ('frame_seconds', True),
        ('frame_seconds', '0.08'),
        ('audio_start_seconds', -0.1),
        ('audio_start_seconds', float('nan')),
        ('audio_start_seconds', float('inf')),
        ('audio_start_seconds', False),
    ],
)
def test_invalid_audio_query_grid_is_rejected(field, value):
    kwargs = dict(frame_seconds=0.08, audio_start_seconds=0.0)
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        map_asr_word_start_anchors(None, '', [], [], 1.0, **kwargs)


def _anchor(frame, start, end):
    return dict(frame_index=frame, token_start=start, token_end=end)


def _tokenizer(offsets, local_ids=None):
    local_ids = list(range(1, len(offsets) + 1)) if local_ids is None else local_ids
    native = Mock(return_value=dict(input_ids=local_ids, offset_mapping=offsets))
    aggregate = SimpleNamespace(
        tokenizers={'native': native},
        tokenizer_offsets={'native': 17},
        tokens=[''] * (18 + max(local_ids)),
        encode=Mock(),
        decode=Mock(),
    )
    return TTSCtcTokenizer(aggregate, 'native'), [token + 17 for token in local_ids], native
