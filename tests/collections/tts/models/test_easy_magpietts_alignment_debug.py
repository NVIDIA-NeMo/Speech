# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import MethodType
from unittest.mock import patch

import pytest
import torch

from nemo.collections.tts.models.easy_magpietts_inference import EasyMagpieTTSInferenceModel
from tests.collections.tts.models.test_easy_magpietts_aligned_inference import _InferenceHarness, _state
from tests.collections.tts.models.test_easy_magpietts_attention_alignment import _config, _make_attention_model
from tests.collections.tts.models.test_easy_magpietts_attention_inference import (
    _fixed_audio_prediction,
    _init_stream,
    _model,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('use_cfg', [False, True])
def test_recording_attention_preserves_cached_outputs_and_captures_conditional_rows(use_cfg):
    baseline = _model({'enabled': False})
    observed = _model({'enabled': False, 'record_alignments': True})
    observed.load_state_dict(baseline.state_dict())
    states = [_init_stream(model, use_cfg=use_cfg) for model in (baseline, observed)]
    initial_position = states[1].cache_seq_len
    tokens = [torch.tensor([7, 10]), torch.tensor([8, 11]), torch.tensor([9, 12]), None]
    for token in tokens:
        outputs = [model.streaming_step(state, token) for model, state in zip((baseline, observed), states)]
        states = [output[0] for output in outputs]
        torch.testing.assert_close(states[0].last_hidden, states[1].last_hidden, rtol=0, atol=0)
        if outputs[0][1] is not None:
            torch.testing.assert_close(outputs[0][1], outputs[1][1], rtol=0, atol=0)

    assert states[0].alignment_debug_history is None
    debug = observed._finalize_alignment_debug(states[1])
    assert len(debug) == 2
    for row, data in enumerate(debug):
        assert data['guidance_source'] == 'none'
        assert data['turns'][0]['source_token_ids'] == ([7, 8, 9] if row == 0 else [10, 11, 12])
        assert data['turns'][0]['source_key_positions'] == list(range(initial_position, initial_position + 3))
        assert data['frames'][0]['start'] == 0
        assert len(data['frames']) == 2  # Two speech-delay steps are absent.
        for frame in data['frames']:
            assert 0 <= frame['attention']['attention_token_index'] < 3
            assert 0 < frame['attention']['attention_probability'] <= 1
            assert frame['text_head'] is None


@pytest.mark.parametrize('use_cfg', [False, True])
def test_head_guidance_advances_next_step_without_enabling_text_feedback(use_cfg):
    model = _head_model({'enabled': True, 'source': 'text_head', 'record_alignments': True})
    state = _init_stream(model, use_cfg=use_cfg)

    def head_prediction(current_state, return_logits=False):
        logits = torch.full((2, model.text_vocab_size), -50.0, device=current_state.config.device, dtype=torch.float32)
        targets = torch.tensor([8, 11], device=logits.device)
        logits[torch.arange(2, device=logits.device), targets] = 50.0
        return (targets, logits) if return_logits else targets

    with (
        patch.object(model, '_predict_aligned_text_tokens', side_effect=head_prediction),
        patch.object(model, 'embed_aligned_text_tokens', wraps=model.embed_aligned_text_tokens) as feedback,
    ):
        for token in (torch.tensor([7, 10]), torch.tensor([8, 11]), torch.tensor([9, 12])):
            state, _, _ = model.streaming_step(state, token)
        controller = state.attention_alignment_inference_state
        assert controller.anchors == [1, 1]
        assert all(observation['anchor_before'] == 0 for observation in controller.last_observations)
        assert all(observation['anchor'] == 1 for observation in controller.last_head_observations)
        state, _, _ = model.streaming_step(state, None)
        assert all(observation['anchor_before'] == 1 for observation in controller.last_observations)
        assert feedback.call_count == 0
        assert not model.aligned_text_feedback

    result = model.streaming_finalize(state)
    assert result.alignment_debug[0]['frames'][0]['text_head']['token_id'] == 8
    assert result.alignment_debug[1]['frames'][0]['text_head']['token_id'] == 11
    assert result.aligned_text_tokens.tolist() == [[8, 8], [11, 11]]


def test_recording_raw_head_preserves_predictions_and_saves_probabilities():
    baseline = _head_model({'enabled': False})
    observed = _head_model({'enabled': False, 'record_alignments': True})
    observed.load_state_dict(baseline.state_dict())
    states = [_init_stream(model, use_cfg=True) for model in (baseline, observed)]
    for token in (torch.tensor([7, 10]), torch.tensor([8, 11]), torch.tensor([9, 12]), None):
        outputs = [model.streaming_step(state, token) for model, state in zip((baseline, observed), states)]
        states = [output[0] for output in outputs]
        torch.testing.assert_close(states[0].last_hidden, states[1].last_hidden, rtol=0, atol=0)
        if outputs[0][1] is not None:
            torch.testing.assert_close(outputs[0][1], outputs[1][1], rtol=0, atol=0)
    torch.testing.assert_close(
        torch.stack(states[0].all_aligned_text_predictions), torch.stack(states[1].all_aligned_text_predictions)
    )
    debug = observed._finalize_alignment_debug(states[1])
    for row, trace in enumerate(debug):
        for index, frame in enumerate(trace['frames']):
            assert frame['text_head']['token_id'] == int(states[1].all_aligned_text_predictions[index][row].item())
            assert 0 < frame['text_head']['probability'] <= 1
    assert not observed.aligned_text_feedback


def test_response_warmup_registers_initial_bpe_keys_without_guiding_silent_steps():
    model = _model({'enabled': True, 'record_alignments': True})
    state = _init_stream(model)
    initial_position = state.cache_seq_len
    with patch.object(model.decoder, 'forward', wraps=model.decoder.forward) as decoder:
        for index in range(5):
            state, audio, _ = model.streaming_step(
                state,
                torch.tensor([7 + index, 20 + index]),
                prefill_like_step=True,
                response_text_warmup=True,
                prefill_like_is_last_step=index == 4,
            )
            assert audio is None
            assert decoder.call_args.kwargs['attention_alignment_inference'] is None
    controller = state.attention_alignment_inference_state
    assert controller.text_key_positions == [list(range(initial_position, initial_position + 5))] * 2
    assert controller.text_token_ids == [list(range(7, 12)), list(range(20, 25))]
    assert state.alignment_debug_history == [[], []]
    state, _, _ = model.streaming_step(state, torch.tensor([12, 25]))
    assert len(state.alignment_debug_history[0]) == 1
    assert state.alignment_debug_history[0][0]['start_frame'] == 5 * model.frame_stacking_factor


def test_turn_reset_archives_source_positions_and_does_not_mix_repeated_ids_between_turns():
    model = _model({'record_alignments': True})
    state = _init_stream(model)
    for token in (torch.tensor([7, 10]), torch.tensor([7, 10]), torch.tensor([8, 11])):
        state, _, _ = model.streaming_step(state, token)
    first_cache = state.cache_seq_len
    model.reset_attention_alignment_turn(state, [0])
    assert state.alignment_debug_turns[0][0]['source_token_ids'] == [7, 7, 8]
    assert state.alignment_debug_turns[0][1]['source_token_ids'] == []
    assert len(state.alignment_debug_turns[1]) == 1
    state, _, _ = model.streaming_step(state, torch.tensor([7, 12]))
    debug = model._finalize_alignment_debug(state)
    assert debug[0]['turns'][1]['source_token_ids'] == [7]
    assert debug[0]['turns'][1]['source_key_positions'] == [first_cache]
    assert [frame['turn_index'] for frame in debug[0]['frames']] == [0, 1]
    assert debug[1]['turns'][0]['source_token_ids'] == [10, 10, 11, 12]


def test_debug_times_remove_bulk_prefill_delay_and_clip_partial_stacked_eos():
    model, state = _InferenceHarness(), _state(batch_size=3)
    model._finalize_alignment_debug = MethodType(EasyMagpieTTSInferenceModel._finalize_alignment_debug, model)
    model.codec_model_samples_per_frame = 10
    model.sample_rate = model.output_sample_rate = 100
    model.aligned_text_use_normalized_text = False
    state.all_predictions = [torch.zeros(3, 1, 6)] + [torch.zeros(3, 1, 2) for _ in range(3)]
    state.audio_prediction_start_idx = torch.tensor([6, 8, -1])
    state.audio_prediction_end_idx = torch.tensor([11, 10, -1])
    state.attention_alignment_inference_state = type(
        'Controller', (), {'enabled': False, 'source': 'attention', 'layer': 'middle', 'head': 0}
    )()
    source = {'turn_index': 0, 'source_token_ids': [7, 7, 8], 'source_key_positions': [12, 13, 14]}
    state.alignment_debug_turns = [[source] for _ in range(3)]
    state.alignment_debug_history = [
        [
            {
                'start_frame': start,
                'end_frame': start + 2,
                'decoder_step': start // 2,
                'turn_index': 0,
                'text_head': {'token_id': token, 'probability': 0.8},
                'attention': {'attention_token_index': index, 'attention_probability': 0.5},
            }
            for start, token, index in [(6, 7, 0), (8, 7, 1), (10, 8, 2)]
        ]
        for _ in range(3)
    ]
    result = model._finalize_alignment_debug(state)
    assert result[0]['audio_frame_count'] == 5
    assert result[0]['audio_duration_seconds'] == pytest.approx(0.5)
    assert [value for frame in result[0]['frames'] for value in (frame['start'], frame['end'])] == pytest.approx(
        [0.0, 0.2, 0.2, 0.4, 0.4, 0.5]
    )
    assert [(frame['start_frame'], frame['end_frame']) for frame in result[1]['frames']] == [(0, 2)]
    assert result[1]['audio_duration_seconds'] == pytest.approx(0.2)
    assert result[2]['frames'] == [] and result[2]['audio_frame_count'] == 0
    assert result[0]['frame_step_seconds'] == pytest.approx(0.2)
    assert result[0]['turns'][0]['source_token_ids'] == [7, 7, 8]
    clipped = model._finalize_alignment_debug(state, torch.tensor([35, 5, 0]))
    assert [frame['end'] for frame in clipped[0]['frames']] == pytest.approx([0.2, 0.35])
    assert clipped[1]['frames'][0]['end'] == pytest.approx(0.05)
    assert clipped[2]['frames'] == []


@pytest.mark.parametrize('eos_subframe', [0, 1])
@torch.no_grad()
def test_multiturn_end_reset_preserves_recorded_partial_stacked_eos_after_bulk_prefill(eos_subframe):
    model = _head_model({'record_alignments': True}, {'frame_stacking_factor': 2})
    state = _init_stream(model)
    state = model.streaming_prefill(state, torch.full((2, 3), model.pad_id, device=state.config.device))
    prefill_frames = sum(prediction.size(-1) for prediction in state.all_predictions)
    assert prefill_frames == 6

    def eos_audio(current_state):
        codes = torch.zeros(
            current_state.config.batch_size,
            model.num_audio_codebooks,
            model.frame_stacking_factor,
            dtype=torch.long,
            device=current_state.config.device,
        )
        codes[:, :, eos_subframe] = model.audio_eos_id
        codes = codes.reshape(current_state.config.batch_size, -1)
        return codes, codes

    model._predict_audio_codes = eos_audio
    for token in (torch.tensor([7, 10]), torch.tensor([8, 11]), torch.tensor([9, 12])):
        state, _, _ = model.streaming_step(state, token)
    assert state.finished.all()
    assert state.audio_prediction_end_idx.tolist() == [prefill_frames + eos_subframe] * 2
    assert [history[-1]['end_frame'] for history in state.alignment_debug_history] == [
        prefill_frames + eos_subframe
    ] * 2

    # The multiturn runner retains the full stacked codec chunk for its joint
    # decode, then clears the turn's EOS index before producing another turn.
    with torch.inference_mode():
        state.audio_prediction_end_idx.fill_(-1)
        state.finished.zero_()
    debug = model._finalize_alignment_debug(state)
    for data in debug:
        assert data['audio_frame_count'] == 2
        if eos_subframe == 0:
            assert data['frames'] == []
        else:
            assert len(data['frames']) == 1
            assert data['frames'][0]['start_frame'] == 0
            assert data['frames'][0]['end_frame'] == 1
            assert data['frames'][0]['end'] == pytest.approx(model.codec_model_samples_per_frame / model.sample_rate)


@pytest.mark.parametrize(
    'extra,message',
    [
        ({'use_aligned_text_head': False}, 'trained use_aligned_text_head'),
        ({'aligned_text_use_normalized_text': True}, 'raw matching BPE'),
        ({'aligned_text_strip_punctuation': True}, 'raw matching BPE'),
    ],
)
def test_text_head_guidance_requires_compatible_trained_raw_head(extra, message):
    with pytest.raises(ValueError, match=message):
        _head_model({'enabled': True, 'source': 'text_head'}, extra)


def _head_model(runtime, extra=None):
    values = {
        'use_multiturn_dataset': True,
        'use_aligned_text_head': True,
        'aligned_text_use_normalized_text': False,
        'aligned_text_strip_punctuation': False,
        'aligned_text_feedback': False,
        'aligned_text_head_dim': 8,
        'attention_alignment_inference': runtime,
    }
    values.update(extra or {})
    model = _make_attention_model(_config(enabled=False, extra=values))
    # This training fixture has fixed phoneme IDs but no decode method. The
    # streaming-finalize assertions also exercise its inference-facing API.
    model.phoneme_tokenizer.decode = lambda token_ids: ' '.join(map(str, token_ids))
    model.register_buffer(
        '_codec_sil_codes_buffer',
        torch.zeros(model.num_audio_codebooks, dtype=torch.long, device=next(model.parameters()).device),
        persistent=False,
    )
    model._predict_audio_codes = _fixed_audio_prediction(model)
    return model.eval()
