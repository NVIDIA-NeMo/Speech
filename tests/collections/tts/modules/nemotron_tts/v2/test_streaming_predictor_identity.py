# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch

from nemo.collections.tts.models.nemotron_tts.v2.base import NemotronTTSBase
from nemo.collections.tts.models.nemotron_tts.v2.model import NemotronTTSModel
from tests.collections.tts.modules.nemotron_tts.v2.test_streaming_timestamps_model import _init_stream, _model

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('use_cfg', [False, True])
def test_timestamps_preserve_actual_acoustic_predictor_sampling_and_cache_progress(use_cfg):
    baseline, observed = _model(), _model(enabled=True)
    for model in (baseline, observed):
        # The timing fixture fixes audio codes to isolate clock/index assertions.
        # Restore the actual V2 method here, including every predictor block and
        # its sampling path. Only codec and speaker-context I/O remain stubbed.
        del model._predict_audio_codes
        assert model._predict_audio_codes.__func__ is NemotronTTSBase._predict_audio_codes
        with torch.no_grad():
            for block in model.acoustic_codes_predictor.blocks:
                bias = block.codebook_projection.bias.view(block.num_codes, block.num_output_tokens)
                bias[:, -1].fill_(-10000.0)
        assert not hasattr(model, 'text_aligner') and not hasattr(model, 'aligned_text_head')
    states = [_init_stream(model, use_cfg=use_cfg) for model in (baseline, observed)]
    for state in states:
        state.config.temperature = 0.7
        state.config.topk = 80
    first_audio_step = first_event_step = first_frame_step = None
    streamed_frames = []
    previous_progress = [cache.get_seq_length() for cache in states[0].acoustic_codes_predictor_cache]
    tokens = [7, 8, 9, 10, 11, 12, baseline.eos_id, baseline.eos_id] + [None] * 5
    with patch(
        'nemo.collections.tts.modules.aligner.TinyGlowTTSAligner.forward',
        side_effect=AssertionError('Timestamp recording cannot call the acoustic teacher'),
    ):
        for step, token_id in enumerate(tokens):
            token = None if token_id is None else torch.full((2,), token_id, dtype=torch.long)
            rng_before = _rng_snapshot()
            ordinary = baseline.streaming_step(states[0], token)
            rng_after = _rng_snapshot()
            _restore_rng(rng_before)
            captured = observed.streaming_step(states[1], token)
            _assert_rng_equal(rng_after, _rng_snapshot())
            states = [ordinary[0], captured[0]]
            torch.testing.assert_close(states[0].last_hidden, states[1].last_hidden, rtol=0, atol=0)
            assert (ordinary[1] is None) == (captured[1] is None)
            if ordinary[1] is not None:
                torch.testing.assert_close(ordinary[1], captured[1], rtol=0, atol=0)
                assert not ordinary[1].eq(baseline.audio_eos_id).any()
                first_audio_step = step if first_audio_step is None else first_audio_step
            progress = []
            for left, right in zip(states[0].acoustic_codes_predictor_cache, states[1].acoustic_codes_predictor_cache):
                assert left.get_seq_length() == right.get_seq_length() == states[0].cache_seq_len
                progress.append(left.get_seq_length())
            assert progress == [length + 1 for length in previous_progress]
            previous_progress = progress
            events = observed.drain_timestamp_events(states[1])
            if events:
                first_event_step = step if first_event_step is None else first_event_step
                assert all(not event.get('forced') or event['confidence'] == 0.0 for event in events)
            frames = observed.drain_timestamp_frames(states[1])
            if frames:
                first_frame_step = step if first_frame_step is None else first_frame_step
                assert captured[1] is not None
                assert len(frames) == 2  # One current-query record per conditional batch row.
                assert {frame['batch_index'] for frame in frames} == {0, 1}
                assert all(frame['valid_codec_frames'] == observed.frame_stacking_factor for frame in frames)
                assert all(not frame['forced'] or frame['confidence'] == 0.0 for frame in frames)
                streamed_frames.extend(frames)
            else:
                assert captured[1] is None
            assert baseline.drain_timestamp_events(states[0]) == []
            assert baseline.drain_timestamp_frames(states[0]) == []
            assert not states[0].finished.any() and not states[1].finished.any()
    assert first_audio_step is not None and first_event_step == first_audio_step == first_frame_step
    snapshots = observed.get_timestamps(states[1])
    for row, snapshot in enumerate(snapshots):
        assert snapshot['frames'] == [frame for frame in streamed_frames if frame['batch_index'] == row]
        assert snapshot['frame_count'] == len(snapshot['frames'])
        assert [frame['frame_index'] for frame in snapshot['frames']] == list(range(snapshot['frame_count']))


def test_target_text_warmup_retains_keys_without_emitting_timestamps_for_silence():
    model = _model(enabled=True)
    state = _init_stream(model, use_cfg=True)
    for token_id in [7, 8, 9, 10, 11, 12]:
        state, _, _ = model.streaming_step(state, torch.full((2,), token_id))
    assert model.drain_timestamp_events(state)
    previous_turn = model.get_timestamps(state)[0]['turns'][-1]['turn_index']
    model.reset_timestamp_turn(state)
    model.drain_timestamp_events(state)
    with torch.inference_mode():
        state.turn_text_tokens_seen.zero_()
        state.text_finished.zero_()
        state.audio_steps.zero_()
    for token_id in [13, 14, 15]:
        state, codes, _ = model.streaming_step(
            state, torch.full((2,), token_id), prefill_like_step=True, timestamp_text_warmup=True
        )
        assert codes is None and model.drain_timestamp_events(state) == []
    current = model.get_timestamps(state)[0]
    turn = current['turns'][-1]
    assert turn['turn_index'] == previous_turn + 1
    assert turn['source_token_ids'] == [13, 14, 15]
    assert all(
        record['start'] is None for record in current['timestamps'] if record['turn_index'] == turn['turn_index']
    )
    for token_id in [16, 17]:
        state, codes, _ = model.streaming_step(state, torch.full((2,), token_id))
        assert codes is None and model.drain_timestamp_events(state) == []
    state, codes, _ = model.streaming_step(state, None)
    events = model.drain_timestamp_events(state)
    assert codes is not None and events
    assert all(event['turn_index'] == turn['turn_index'] for event in events)
    assert model.get_timestamps(state)[0]['turns'][-1]['source_token_ids'] == [13, 14, 15, 16, 17]


@pytest.mark.parametrize('use_cfg', [False, True])
@pytest.mark.parametrize('has_user_prefix', [False, True])
def test_user_audio_warmup_preserves_training_audio_inputs(use_cfg, has_user_prefix):
    model = _model()
    del model._predict_audio_codes
    state = _init_stream(model, use_cfg=use_cfg)
    delay = state.config.training_mode.streaming_speech_delay
    offset = int(has_user_prefix)
    codes = torch.zeros(2, 4, delay + offset + 1, dtype=torch.long)
    agent_mask = torch.zeros(2, codes.size(-1), dtype=torch.bool)
    agent_mask[:, -1] = True
    training_audio, *_ = NemotronTTSModel.prepare_audio_channel_embeddings(
        model,
        codes,
        torch.full((2,), codes.size(-1)),
        torch.zeros(2, dtype=torch.long),
        agent_mask=agent_mask,
        audio_codes_are_stacked=True,
    )
    if has_user_prefix:
        with torch.inference_mode():
            state.audio_steps.fill_(2)
            state.last_audio_codes = torch.full((2, 4), model.audio_user_speaking_id)
    token = torch.full((2,), 7)
    text_embedding = model.embed_text_tokens(token[:, None], text_lens=torch.ones(2, dtype=torch.long))
    user_embedding = torch.full((2, model.cfg.embedding_dim), 0.125)
    with patch.object(model, 'forward', wraps=model.forward) as forward:
        for step in range(delay):
            state, audio, _ = model.streaming_step(
                state,
                token,
                user_audio_channel_embedding=user_embedding,
                prefill_like_step=True,
                prefill_like_is_last_step=step == delay - 1,
            )
            assert audio is None
    for step, call in enumerate(forward.call_args_list):
        expected_audio = training_audio[:, step + offset : step + offset + 1]
        expected = text_embedding + expected_audio + user_embedding[:, None]
        if use_cfg:
            expected = torch.cat([expected, expected_audio])
        torch.testing.assert_close(call.kwargs['inputs_embeds'], expected)
    with patch.object(model, 'forward', wraps=model.forward) as forward:
        state, audio, _ = model.streaming_step(state, token)
    assert audio is not None
    expected_audio = training_audio[:, delay + offset : delay + offset + 1]
    expected = text_embedding + expected_audio
    if use_cfg:
        expected = torch.cat([expected, expected_audio])
    torch.testing.assert_close(forward.call_args.kwargs['inputs_embeds'], expected)
    assert all(cache.get_seq_length() == state.cache_seq_len for cache in state.acoustic_codes_predictor_cache)


@pytest.mark.parametrize('use_cfg', [False, True])
@pytest.mark.parametrize('previous_agent_eos', [False, True])
def test_user_prefill_preserves_training_previous_audio_input(use_cfg, previous_agent_eos):
    model = _model()
    state = _init_stream(model, use_cfg=use_cfg)
    codes = torch.zeros(2, 4, 5, dtype=torch.long)
    agent_mask = torch.zeros(2, 5, dtype=torch.bool)
    agent_mask[:, -1] = True
    if previous_agent_eos:
        codes[:, :, 0] = model.audio_eos_id
        agent_mask[:, 0] = True
        state.last_audio_codes = torch.full((2, 4), model.audio_eos_id)
    training_audio, *_ = NemotronTTSModel.prepare_audio_channel_embeddings(
        model,
        codes,
        torch.full((2,), codes.size(-1)),
        torch.zeros(2, dtype=torch.long),
        agent_mask=agent_mask,
        audio_codes_are_stacked=True,
    )
    with patch.object(model, 'forward', wraps=model.forward) as forward:
        state = model.streaming_prefill(state, torch.full((2, 3), model.pad_id))
    offset = int(previous_agent_eos)
    expected = training_audio[:, offset : offset + 3]
    if use_cfg:
        expected = expected.repeat(2, 1, 1)
    torch.testing.assert_close(forward.call_args.kwargs['inputs_embeds'], expected)
    assert all(cache.get_seq_length() == state.cache_seq_len for cache in state.acoustic_codes_predictor_cache)


@pytest.mark.parametrize('use_cfg', [False, True])
def test_user_prefill_matches_serialized_predictor_cache_contents(use_cfg):
    models = [_model(), _model()]
    states = []
    for model in models:
        del model._predict_audio_codes
        model.cfg.condition_on_user_speech = True
        with torch.no_grad():
            for block in model.acoustic_codes_predictor.blocks:
                block.codebook_projection.bias.view(block.num_codes, block.num_output_tokens)[:, -1].fill_(-10000.0)
        state = _init_stream(model, use_cfg=use_cfg)
        state.config.temperature = 0.0
        for token_id in range(7, 14):
            state, _, _ = model.streaming_step(state, torch.full((2,), token_id))
        assert state.all_predictions
        states.append(state)
    user_audio = torch.arange(96, dtype=torch.float32).reshape(2, 3, 16) / 128
    tokens = torch.zeros(2, 3, dtype=torch.long)
    with (
        patch.object(models[0], 'forward', wraps=models[0].forward) as bulk,
        patch.object(models[1], 'forward', wraps=models[1].forward) as serial,
    ):
        states[0] = models[0].streaming_prefill(states[0], tokens, user_audio_channel_embedding=user_audio)
        for step in range(tokens.size(1)):
            states[1], audio, _ = models[1].streaming_step(
                states[1], tokens[:, step], user_audio_channel_embedding=user_audio[:, step], prefill_like_step=True
            )
            assert audio is None
    torch.testing.assert_close(
        bulk.call_args.kwargs['inputs_embeds'],
        torch.cat([call.kwargs['inputs_embeds'] for call in serial.call_args_list], dim=1),
    )
    torch.testing.assert_close(states[0].last_hidden[:, -1:], states[1].last_hidden, rtol=1e-5, atol=1e-6)
    for left, right in zip(states[0].acoustic_codes_predictor_cache, states[1].acoustic_codes_predictor_cache):
        assert left.get_seq_length() == right.get_seq_length() == states[0].cache_seq_len == states[1].cache_seq_len
        for name in ('key_cache', 'value_cache'):
            for left_tensor, right_tensor in zip(getattr(left, name), getattr(right, name)):
                torch.testing.assert_close(left_tensor, right_tensor, rtol=1e-5, atol=1e-6)


def _rng_snapshot():
    return torch.random.get_rng_state().clone(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []


def _restore_rng(snapshot):
    torch.random.set_rng_state(snapshot[0])
    if snapshot[1]:
        torch.cuda.set_rng_state_all(snapshot[1])


def _assert_rng_equal(expected, actual):
    assert torch.equal(expected[0], actual[0])
    assert len(expected[1]) == len(actual[1])
    assert all(torch.equal(left, right) for left, right in zip(expected[1], actual[1]))
