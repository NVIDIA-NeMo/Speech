# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch

from tests.collections.tts.models.test_easy_magpietts import _make_easy_magpie_model, _toy_batch, tiny_easy_magpie_cfg
from tests.collections.tts.models.test_easy_magpietts_attention_alignment import _config, _make_attention_model

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('use_cfg', [False, True])
def test_enabled_guidance_changes_cached_conditional_predictions_and_preserves_cfg_unconditional_branch(use_cfg):
    baseline = _model()
    guided = _model({'enabled': True, 'look_back_tokens': 0, 'look_ahead_tokens': 0})
    guided.load_state_dict(baseline.state_dict())
    states = [_init_stream(model, use_cfg=use_cfg) for model in (baseline, guided)]

    for index, text in enumerate((torch.tensor([7, 10]), torch.tensor([8, 11]), torch.tensor([9, 12]))):
        states = [model.streaming_step(state, text)[0] for model, state in zip((baseline, guided), states)]
        if index < 2:
            torch.testing.assert_close(states[0].last_hidden, states[1].last_hidden, rtol=0, atol=0)
        else:
            assert not torch.equal(states[0].last_hidden[:2], states[1].last_hidden[:2])
            if use_cfg:
                torch.testing.assert_close(states[0].last_hidden[2:], states[1].last_hidden[2:], rtol=0, atol=0)


@pytest.mark.parametrize('use_cfg', [False, True])
def test_disabled_runtime_guidance_preserves_native_cached_stream_bitwise(use_cfg):
    baseline = _model()
    disabled = _model({'enabled': False})
    disabled.load_state_dict(baseline.state_dict())
    states = [_init_stream(model, use_cfg=use_cfg) for model in (baseline, disabled)]
    assert all(state.attention_alignment_inference_state is None for state in states)
    tokens = [torch.tensor([7, 10]), torch.tensor([8, 11]), torch.tensor([9, 12]), None, None]

    with patch.object(disabled.decoder, 'forward', wraps=disabled.decoder.forward) as run:
        for text in tokens:
            results = [model.streaming_step(state, text) for model, state in zip((baseline, disabled), states)]
            states = [result[0] for result in results]
            torch.testing.assert_close(states[0].last_hidden, states[1].last_hidden, rtol=0, atol=0)
            assert states[0].cache_seq_len == states[1].cache_seq_len
            assert run.call_args.kwargs.get('attention_alignment_inference') is None
            if results[0][1] is not None:
                torch.testing.assert_close(results[0][1], results[1][1], rtol=0, atol=0)

    assert not hasattr(disabled, 'text_aligner')
    assert not hasattr(disabled, 'aligned_text_head')


@pytest.mark.parametrize('use_cfg', [False, True])
def test_runtime_guidance_uses_observed_current_turn_keys_and_conditional_rows_only(use_cfg):
    model = _model({'enabled': True})
    with patch.object(model.decoder, 'forward', wraps=model.decoder.forward) as run:
        state = _init_stream(model, use_cfg=use_cfg)
        assert run.call_args.kwargs.get('attention_alignment_inference') is None
        controller = state.attention_alignment_inference_state
        assert controller.text_key_positions == [[], []]
        start = state.cache_seq_len

        for index, text in enumerate((torch.tensor([7, 10]), torch.tensor([8, 11]), torch.tensor([9, 12]))):
            state, _, _ = model.streaming_step(state, text)
            assert controller.text_key_positions == [list(range(start, start + index + 1))] * 2
            request = run.call_args.kwargs.get('attention_alignment_inference')
            if index < 2:
                assert request is None
            else:
                assert request.state is controller
                assert request.batch_indices == (0, 1)
                assert all(0 <= position < index + 1 for position in controller.last_attended_tokens)
                assert state.last_hidden.size(0) == (4 if use_cfg else 2)

        keys_before = [list(keys) for keys in controller.text_key_positions]
        state, _, _ = model.streaming_step(state, torch.full((2,), model.eos_id))
        assert controller.text_key_positions == keys_before
        assert not state.finished.any(), 'Text EOS must not force audio EOS.'
        assert state.text_finished.all()
        state, _, _ = model.streaming_step(state, None)
        assert controller.text_key_positions == keys_before
        assert run.call_args.kwargs['attention_alignment_inference'].batch_indices == (0, 1)

    assert not hasattr(model, 'text_aligner')


def test_context_and_ignored_tokens_are_not_registered_as_current_turn_bpe_keys():
    model = _model({'enabled': True})
    state = _init_stream(model, ragged=True)
    controller = state.attention_alignment_inference_state
    assert (state.context_position < state.full_context_lens).tolist() == [True, False]
    first_position = state.cache_seq_len

    state, _, _ = model.streaming_step(state, torch.tensor([7, 10]))

    assert controller.text_key_positions == [[], [first_position]]
    second_position = state.cache_seq_len
    state, _, _ = model.streaming_step(state, torch.tensor([8, model.pad_id]))
    assert controller.text_key_positions == [[second_position], [first_position]]


def test_dropped_text_suppresses_guidance_even_after_valid_keys_were_registered():
    model = _model({'enabled': True})
    state = _init_stream(model)
    for text in (torch.tensor([7, 10]), torch.tensor([8, 11])):
        state, _, _ = model.streaming_step(state, text)
    keys_before = [list(keys) for keys in state.attention_alignment_inference_state.text_key_positions]

    with patch.object(model.decoder, 'forward', wraps=model.decoder.forward) as run:
        state, _, _ = model.streaming_step(state, torch.tensor([9, 12]), force_dropout_text=True)

    assert run.call_args.kwargs.get('attention_alignment_inference') is None
    assert state.attention_alignment_inference_state.text_key_positions == keys_before


@pytest.mark.parametrize('profile_mode', ['bulk', 'step'])
def test_user_profile_prefill_clears_old_turn_and_never_registers_user_keys(profile_mode):
    model = _model({'enabled': True})
    state = _init_stream(model, use_cfg=True)
    for text in (torch.tensor([7, 10]), torch.tensor([8, 11]), torch.tensor([9, 12])):
        state, _, _ = model.streaming_step(state, text)
    controller = state.attention_alignment_inference_state
    assert all(controller.text_key_positions)

    with patch.object(model.decoder, 'forward', wraps=model.decoder.forward) as run:
        if profile_mode == 'bulk':
            state = model.streaming_prefill(state, torch.tensor([[13, 14], [15, 16]]))
        else:
            state, _, _ = model.streaming_step(
                state, torch.tensor([13, 15]), prefill_like_step=True, prefill_like_is_last_step=True
            )
        assert run.call_args.kwargs.get('attention_alignment_inference') is None

    assert controller.text_key_positions == [[], []]
    assert controller.anchors == [0, 0]
    assert controller.last_attended_tokens == [-1, -1]
    position = state.cache_seq_len
    state, _, _ = model.streaming_step(state, torch.tensor([17, 18]))
    assert controller.text_key_positions == [[position], [position]]


def test_explicit_turn_reset_can_clear_only_one_conversation_in_a_batch():
    model = _model({'enabled': True})
    state = _init_stream(model)
    for text in (torch.tensor([7, 10]), torch.tensor([8, 11]), torch.tensor([9, 12])):
        state, _, _ = model.streaming_step(state, text)
    controller = state.attention_alignment_inference_state
    second_keys = list(controller.text_key_positions[1])
    second_anchor = controller.anchors[1]
    second_attended = controller.last_attended_tokens[1]

    model.reset_attention_alignment_turn(state, batch_indices=[0])

    assert controller.text_key_positions[0] == []
    assert controller.anchors[0] == 0
    assert controller.last_attended_tokens[0] == -1
    assert controller.text_key_positions[1] == second_keys
    assert controller.anchors[1] == second_anchor
    assert controller.last_attended_tokens[1] == second_attended


def test_new_text_bos_resets_only_that_response_turn():
    model = _model({'enabled': True})
    state = _init_stream(model)
    for text in (torch.tensor([7, 10]), torch.tensor([8, 11]), torch.tensor([9, 12])):
        state, _, _ = model.streaming_step(state, text)
    controller = state.attention_alignment_inference_state
    second_keys = list(controller.text_key_positions[1])
    position = state.cache_seq_len

    state, _, _ = model.streaming_step(state, torch.tensor([model.bos_id, 13]))

    assert controller.text_key_positions[0] == []
    assert controller.last_attended_tokens[0] == -1
    assert controller.text_key_positions[1] == second_keys + [position]


def test_no_text_keys_means_no_guidance_request_during_audio_generation():
    model = _model({'enabled': True})
    state = _init_stream(model)
    with patch.object(model.decoder, 'forward', wraps=model.decoder.forward) as run:
        for _ in range(4):
            state, _, _ = model.streaming_step(state, None)

    assert state.attention_alignment_inference_state.text_key_positions == [[], []]
    assert run.call_args.kwargs.get('attention_alignment_inference') is None


@pytest.mark.parametrize(
    'settings,message',
    [
        ({'enabled': 'false'}, 'enabled'),
        ({'enabled': True, 'look_back_tokens': -1}, 'look_back_tokens'),
        ({'enabled': True, 'look_ahead_tokens': True}, 'look_ahead_tokens'),
        ({'enabled': True, 'prior_epsilon': 0.0}, 'prior_epsilon'),
        ({'enabled': True, 'prior_epsilon': 1.1}, 'prior_epsilon'),
        ({'enabled': True, 'head': 4}, 'head'),
        ({'enabled': True, 'layer': 3}, 'layer'),
    ],
)
def test_invalid_runtime_guidance_configuration_fails_during_model_initialization(settings, message):
    with pytest.raises(ValueError, match=message):
        _model(settings)


def test_ipa_input_cannot_be_misregistered_as_a_bpe_alignment_key():
    model = _make_attention_model(
        _config(
            enabled=False,
            extra={
                'use_multiturn_dataset': True,
                'enable_phoneme_text_input': True,
                'attention_alignment_inference': {'enabled': True},
            },
        )
    )
    model._predict_audio_codes = _fixed_audio_prediction(model)
    state = _init_stream(model)
    ipa_token = model.text_phoneme_token_offset + 4

    with patch.object(model.decoder, 'forward', wraps=model.decoder.forward) as run:
        for _ in range(3):
            state, _, _ = model.streaming_step(state, torch.full((2,), ipa_token))

    assert state.attention_alignment_inference_state.text_key_positions == [[], []]
    assert run.call_args.kwargs.get('attention_alignment_inference') is None


def _model(runtime=None):
    values = {
        'use_multiturn_dataset': True,
        'nemotron_h_config': {'num_hidden_layers': 5, 'hybrid_override_pattern': '*-*-*'},
    }
    if runtime is not None:
        values['attention_alignment_inference'] = runtime
    model = _make_easy_magpie_model(tiny_easy_magpie_cfg(values))
    # The random tiny codec fixture skips production silence-token calibration.
    # Supply valid codec IDs so native user-profile prefill can run unchanged.
    model.register_buffer(
        '_codec_sil_codes_buffer',
        torch.zeros(model.num_audio_codebooks, dtype=torch.long, device=next(model.parameters()).device),
        persistent=False,
    )
    model._predict_audio_codes = _fixed_audio_prediction(model)
    return model.eval()


def _init_stream(model, *, use_cfg=False, ragged=False):
    batch = _toy_batch(model)
    if not ragged:
        batch['context_audio_codes_lens'].fill_(batch['context_audio_codes'].size(-1))
    return model.streaming_init(
        context_audio_codes=batch['context_audio_codes'],
        context_audio_codes_lens=batch['context_audio_codes_lens'],
        context_text_tokens=batch['context_text_tokens'],
        context_text_tokens_lens=batch['context_text_tokens_lens'],
        use_cfg=use_cfg,
        cfg_scale=1.5 if use_cfg else 1.0,
        temperature=1.0,
        topk=1,
    )


def _fixed_audio_prediction(model):
    def predict(state):
        codes = torch.zeros(
            state.config.batch_size,
            model.num_audio_codebooks * model.frame_stacking_factor,
            dtype=torch.long,
            device=state.config.device,
        )
        return codes, codes

    return predict
