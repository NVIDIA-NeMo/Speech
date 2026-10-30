# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch
import torch.nn.functional as F

from nemo.collections.tts.modules.attention_alignment_inference import (
    AttentionAlignmentInferenceState,
    apply_attention_alignment_inference,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('source', ['attention', 'text_head'])
def test_record_only_attention_is_bitwise_identical_to_ordinary_cached_output(source):
    state = _state(source=source, enabled=False, record_alignments=True, look_ahead_tokens=1)
    ordinary, output = _apply(state, [0.0, 1.0, 5.0])
    assert torch.equal(output, ordinary)
    assert state.anchors == [0]
    observation = state.last_observations[0]
    assert observation['attention_token_index'] == observation['guided_attention_token_index'] == 2
    assert observation['attention_probability'] == observation['guided_attention_probability']
    state.observe_text_logits(_logits(5), ignored_token_ids=(0, 7))
    assert state.anchors == [0]
    assert state.last_head_observations[0]['token_id'] == 5


def test_text_head_updates_the_next_window_without_using_attention_to_advance():
    state = _state(source='text_head', look_back_tokens=0, look_ahead_tokens=1)
    _, before = _apply(state, [0.0, 0.0, 0.0])
    assert state.anchors == [0]
    torch.testing.assert_close(before[0, 0, 0], torch.tensor([1.0, 1.0, 0.1]) / 2.1)
    state.observe_text_logits(_logits(5), ignored_token_ids=(0, 7))
    assert state.anchors == [1]
    assert state.last_head_observations[0]['accepted']
    _, after = _apply(state, [0.0, 0.0, 0.0])
    torch.testing.assert_close(after[0, 0, 0], torch.tensor([0.1, 1.0, 1.0]) / 2.1)
    assert state.anchors == [1], 'The attention peak must not advance a text-head controller.'


@pytest.mark.parametrize(
    'settings,probabilities,reason',
    [
        ({'min_confidence': 0.8}, [0.0, 0.0, 0.0, 0.0, 0.35, 0.6, 0.05, 0.0], 'low_confidence'),
        ({'min_margin': 0.3}, [0.0, 0.0, 0.0, 0.0, 0.4, 0.6, 0.0, 0.0], 'low_margin'),
        ({'min_token_frames': 2}, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 'minimum_frames'),
    ],
)
def test_text_head_progress_honors_confidence_margin_and_minimum_frame_thresholds(settings, probabilities, reason):
    state = _state(source='text_head', **settings)
    _apply(state)
    logits = torch.tensor(probabilities).log().unsqueeze(0)
    state.observe_text_logits(logits)
    assert state.anchors == [0]
    assert state.last_head_observations[0]['reason'] == reason
    if reason == 'minimum_frames':
        state.observe_text_logits(logits)
        assert state.anchors == [1]


@pytest.mark.parametrize('token_id,reason', [(0, 'ignored_token'), (7, 'ignored_token'), (3, 'unmatched_token')])
def test_pad_eos_and_unmatched_predictions_hold_progress_without_terminating(token_id, reason):
    state = _state(source='text_head')
    _apply(state)
    state.observe_text_logits(_logits(token_id), ignored_token_ids=(0, 7))
    assert state.anchors == [0]
    assert state.last_head_observations[0]['reason'] == reason
    assert not state.last_head_observations[0]['accepted']


def test_adjacent_identical_input_ids_are_marked_ambiguous_and_do_not_invent_a_boundary():
    state = _state(source='text_head', token_ids=(4, 4, 5))
    _apply(state)
    state.observe_text_logits(_logits(4))
    observation = state.last_head_observations[0]
    assert observation['ambiguous'] and observation['reason'] == 'ambiguous_token'
    assert observation['candidate_token_index'] == 0
    assert state.anchors == [0]
    # A later distinct token can establish progress, but the head did not locate
    # the internal boundary between the two adjacent occurrences of token 4.
    state.observe_text_logits(_logits(5))
    assert state.anchors == [2]


def test_head_uses_earliest_forward_occurrence_and_never_moves_backwards():
    state = _state(source='text_head', token_ids=(4, 5, 5))
    _apply(state)
    state.observe_text_logits(_logits(5))
    assert state.anchors == [1]
    assert state.last_head_observations[0]['ambiguous']
    state.observe_text_logits(_logits(4))
    assert state.anchors == [1]
    assert state.last_head_observations[0]['reason'] == 'unmatched_token'


def test_head_observation_matches_the_output_argmax_when_two_vocabulary_logits_tie():
    state = _state(source='text_head')
    _apply(state)
    logits = _logits(4)
    logits[0, 5] = logits[0, 4]
    state.observe_text_logits(logits)
    assert state.last_head_observations[0]['token_id'] == int(logits.argmax(dim=-1).item()) == 4
    assert state.anchors == [0]


@pytest.mark.parametrize('masked', [False, True])
def test_head_never_matches_future_or_masked_conditioning_keys(masked):
    state = _state(source='text_head')
    mask = torch.tensor([[True, not masked, True]]) if masked else None
    _apply(state, cache_position=1, attention_mask=mask)
    state.observe_text_logits(_logits(6))
    assert state.anchors == [0]
    assert state.last_head_observations[0]['reason'] == 'unmatched_token'
    if masked:
        state.observe_text_logits(_logits(5))
        assert state.anchors == [0]


def test_attention_thresholds_use_full_key_probability_instead_of_text_only_normalization():
    state = AttentionAlignmentInferenceState(1, min_confidence=0.2, look_ahead_tokens=1)
    state.add_text_token(0, 0, 4)
    state.add_text_token(0, 1, 5)
    _apply(state, [0.0, 10.0, 100.0])
    assert state.last_observations[0]['attention_token_index'] == 1
    assert state.last_observations[0]['text_attention_mass'] < 0.01
    assert state.anchors == [0]
    assert not state.last_observations[0]['accepted']


def test_attention_margin_and_minimum_frames_hold_an_uncertain_or_premature_advance():
    state = _state(min_margin=0.4, look_ahead_tokens=1)
    _apply(state, [0.0, 0.1, -10.0])
    assert state.anchors == [0]
    state = _state(min_token_frames=2, look_ahead_tokens=1)
    _apply(state, [0.0, 10.0, -10.0])
    assert state.anchors == [0]
    _apply(state, [0.0, 10.0, -10.0])
    assert state.anchors == [1]


def test_begin_step_and_turn_reset_clear_stale_diagnostics_and_source_ids():
    state = _state(source='text_head')
    _apply(state)
    state.observe_text_logits(_logits(5))
    assert state.anchors == [1]
    state.begin_step()
    assert state.last_observations == [None] and state.last_head_observations == [None]
    state.observe_text_logits(_logits(6))
    assert state.anchors == [1]
    assert state.last_head_observations[0]['reason'] == 'unmatched_token'
    state.reset_turn(0)
    assert state.text_token_ids == [[]]
    assert state.text_key_positions == [[]]
    assert state.anchors == [0] and state.last_head_observations == [None]


def test_invalid_head_logits_hold_progress_and_keep_diagnostics_strict_json_serializable():
    state = _state(source='text_head')
    _apply(state)
    state.observe_text_logits(torch.full((1, 8), -torch.inf))
    assert state.anchors == [0]
    observation = state.last_head_observations[0]
    assert observation['reason'] == 'invalid_logits' and not observation['accepted']
    assert observation['probability'] == observation['margin'] == 0.0
    json.dumps(observation, allow_nan=False)


@pytest.mark.parametrize(
    'settings',
    [
        {'source': 'mas'},
        {'enabled': 1},
        {'record_alignments': None},
        {'min_confidence': -0.1},
        {'min_confidence': True},
        {'min_confidence': float('nan')},
        {'min_margin': 1.1},
        {'min_margin': '0.1'},
        {'min_token_frames': 0},
        {'min_token_frames': 1.0},
    ],
)
def test_new_controller_settings_are_validated(settings):
    with pytest.raises(ValueError):
        AttentionAlignmentInferenceState(1, **settings)


def _state(token_ids=(4, 5, 6), **kwargs):
    state = AttentionAlignmentInferenceState(1, **kwargs)
    for position, token_id in enumerate(token_ids):
        state.add_text_token(0, position, token_id)
    return state


def _apply(state, scores=None, cache_position=None, attention_mask=None):
    scores = [0.0, 1.0, 2.0] if scores is None else scores
    size = len(scores)
    query = torch.ones(1, 1, 1, size)
    key = torch.zeros(1, 1, size, size)
    key[0, 0, :, 0] = torch.tensor(scores) * size**0.5
    value = torch.eye(size).reshape(1, 1, size, size)
    position = size - 1 if cache_position is None else cache_position
    visible = torch.arange(size) <= position
    if attention_mask is not None:
        visible &= attention_mask[0]
    ordinary = F.scaled_dot_product_attention(query, key, value, attn_mask=visible.reshape(1, 1, 1, size))
    output = apply_attention_alignment_inference(
        query,
        key,
        value,
        ordinary,
        state.request(),
        cache_position=torch.tensor([position]),
        attention_mask=attention_mask,
    )
    return ordinary, output


def _logits(token_id):
    logits = torch.full((1, 8), -10.0)
    logits[0, token_id] = 10.0
    return logits
