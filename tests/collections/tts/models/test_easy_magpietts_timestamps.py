# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch

from nemo.collections.tts.modules import nemotron_h_decoder
from tests.collections.tts.models.test_easy_magpietts_alignment_debug import _head_model
from tests.collections.tts.models.test_easy_magpietts_attention_inference import _init_stream, _model

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('use_cfg', [False, True])
def test_timestamp_only_preserves_audio_and_hidden_states_without_guidance_or_debug_history(use_cfg):
    baseline = _model({'enabled': False})
    timestamped = _model({'enabled': False, 'record_timestamps': True})
    timestamped.load_state_dict(baseline.state_dict())
    states = [_init_stream(model, use_cfg=use_cfg, ragged=True) for model in (baseline, timestamped)]
    source = torch.tensor([[7, 8, 9], [10, 11, 12]], device=states[0].config.device)
    for _ in range(10):
        positions = states[0].text_tokens_seen
        token = source[torch.arange(2, device=source.device), positions.clamp(max=2)]
        token = torch.where(positions < 3, token, torch.full_like(token, baseline.eos_id))
        outputs = [model.streaming_step(state, token) for model, state in zip((baseline, timestamped), states)]
        states = [output[0] for output in outputs]
        torch.testing.assert_close(states[0].last_hidden, states[1].last_hidden, rtol=0, atol=0)
        if outputs[0][1] is not None:
            torch.testing.assert_close(outputs[0][1], outputs[1][1], rtol=0, atol=0)
    state = states[1]
    assert state.alignment_debug_history is None and state.alignment_debug_turns is None
    assert states[0].timestamp_turns is None
    assert not hasattr(timestamped, 'text_aligner')
    assert not hasattr(timestamped, 'aligned_text_head')
    assert state.attention_alignment_inference_state.anchors == [0, 0]
    traces = timestamped.get_timestamps(state)
    assert traces[0]['source'] == 'attention' and not traces[0]['guidance_enabled']
    assert [token['token_id'] for token in traces[0]['timestamps']] == [7, 8, 9]
    assert [token['token_id'] for token in traces[1]['timestamps']] == [10, 11, 12]
    assert all(
        len(observation['probabilities']) <= 3
        for observation in state.attention_alignment_inference_state.last_raw_attention
        if observation is not None
    )


def test_timestamps_match_output_query_without_bos_token_or_end_token_shift_at_delay_five_and_stack_two():
    model = _timestamp_model(confirmation_frames=1)
    state = _init_stream(model)
    state.config.training_mode.streaming_speech_delay = 5
    # BOS and EOS are timeline keys, not input-token occurrences. Equal BPE IDs
    # must retain separate positional identities through the last source token.
    initial = [model.bos_id, 7, 7, 8, model.eos_id]
    with _attention_peaks([0, 0, 1, 2, 2]):
        for token in initial:
            state, audio, _ = model.streaming_step(state, torch.full((2,), token))
            assert audio is None
        for _ in range(5):
            state, _, _ = model.streaming_step(state, None)
    traces = model.get_timestamps(state, finalize=True)
    step = model.frame_stacking_factor * model.codec_model_samples_per_frame / model.sample_rate
    for trace in traces:
        entries = trace['timestamps']
        assert [entry['token_id'] for entry in entries] == [7, 7, 8]
        assert [entry['token_index'] for entry in entries] == [0, 1, 2]
        assert [value for entry in entries for value in (entry['start'], entry['end'])] == pytest.approx(
            [0.0, 2 * step, 2 * step, 3 * step, 3 * step, 5 * step]
        )
        assert trace['frame_step_seconds'] == pytest.approx(step)


def test_training_stacked_bos_is_separate_from_first_two_real_codec_targets():
    model = _timestamp_model(confirmation_frames=1)
    device = next(model.parameters()).device
    codes = torch.arange(4, device=device).reshape(1, 1, 4).expand(2, model.num_audio_codebooks, 4).clone()
    lengths = torch.full((2,), 4, device=device, dtype=torch.long)
    delay = torch.full((2,), 5, device=device, dtype=torch.long)
    with patch.object(model, 'embed_audio_tokens', wraps=model.embed_audio_tokens) as embed:
        _, _, targets, _, _ = model.prepare_audio_channel_embeddings(codes, lengths, delay)
    inputs = embed.call_args.args[0]
    assert inputs[:, :, 0].eq(model.audio_bos_id).all()
    assert targets[:, :, 0].reshape(2, model.num_audio_codebooks, 2).tolist() == codes[:, :, :2].tolist()
    assert torch.equal(inputs[:, :, 1], targets[:, :, 0])
    assert targets[:, :, 1].reshape(2, model.num_audio_codebooks, 2).tolist() == codes[:, :, 2:4].tolist()


def test_timestamp_boundaries_are_available_during_streaming_and_drained_per_batch():
    model = _timestamp_model(confirmation_frames=2)
    state = _init_stream(model)
    with _attention_peaks([0, 0, 1, 1]):
        for token in (torch.tensor([7, 10]), torch.tensor([8, 11]), torch.tensor([9, 12]), None, None, None):
            state, _, _ = model.streaming_step(state, token)
    first_row_events = model.drain_timestamp_events(state, [0])
    assert first_row_events and all(event['batch_index'] == 0 for event in first_row_events)
    assert all(event['turn_index'] == 0 for event in first_row_events)
    assert any(event['token_index'] == 1 and event['event'] == 'confirmed' for event in first_row_events)
    assert any(event['latency_seconds'] > 0 for event in first_row_events)
    remaining = model.drain_timestamp_events(state)
    assert remaining and all(event['batch_index'] == 1 for event in remaining)
    assert model.drain_timestamp_events(state) == []
    assert not state.finished.any(), 'Timestamp collection must not finish the audio stream.'


@pytest.mark.parametrize('eos_subframe', [0, 1])
def test_timestamp_eos_clips_partial_stacked_frame_and_survives_multiturn_end_reset(eos_subframe):
    model = _timestamp_model(confirmation_frames=1)
    state = _init_stream(model)
    state = model.streaming_prefill(state, torch.full((2, 3), model.pad_id, device=state.config.device))
    assert sum(prediction.size(-1) for prediction in state.all_predictions) == 6

    def eos_audio(current_state):
        codes = torch.zeros(2, model.num_audio_codebooks, 2, dtype=torch.long, device=current_state.config.device)
        codes[:, :, eos_subframe] = model.audio_eos_id
        return codes.reshape(2, -1), codes.reshape(2, -1)

    model._predict_audio_codes = eos_audio
    with _attention_peaks([0]):
        for token in (torch.tensor([7, 10]), torch.tensor([8, 11]), torch.tensor([9, 12])):
            state, _, _ = model.streaming_step(state, token)
    with torch.inference_mode():
        state.audio_prediction_end_idx.fill_(-1)
        state.finished.zero_()
    traces = model.get_timestamps(state, finalize=True)
    duration = eos_subframe * model.codec_model_samples_per_frame / model.sample_rate
    for trace in traces:
        assert trace['audio_duration_seconds'] == pytest.approx(duration)
        assert trace['timestamps'][0]['end'] == (pytest.approx(duration) if eos_subframe else None)
        assert all(token['end'] is None or token['end'] <= duration for token in trace['timestamps'])


def test_response_warmup_tokens_are_kept_and_turn_reset_archives_only_current_turn():
    model = _timestamp_model(confirmation_frames=1)
    state = _init_stream(model)
    with _attention_peaks([2, 0]):
        for index in range(5):
            state, _, _ = model.streaming_step(
                state,
                torch.tensor([7 + index, 20 + index]),
                prefill_like_step=True,
                response_text_warmup=True,
                prefill_like_is_last_step=index == 4,
            )
        state, _, _ = model.streaming_step(state, torch.tensor([12, 25]))
        first_end = model.get_timestamps(state)[0]['audio_duration_seconds']
        model.reset_attention_alignment_turn(state, [0])
        state, _, _ = model.streaming_step(state, torch.tensor([7, 26]))
    traces = model.get_timestamps(state, finalize=True)
    assert traces[0]['turns'][0]['source_token_ids'] == list(range(7, 13))
    assert traces[0]['turns'][1]['source_token_ids'] == [7]
    assert traces[1]['turns'][0]['source_token_ids'] == list(range(20, 27))
    new_token = next(token for token in traces[0]['timestamps'] if token['turn_index'] == 1)
    assert new_token['start'] == pytest.approx(first_end)


def test_timestamp_only_does_not_request_extra_text_head_logits():
    model = _head_model({'enabled': False, 'source': 'text_head', 'record_timestamps': True})
    state = _init_stream(model)
    with patch.object(model, '_predict_aligned_text_tokens', wraps=model._predict_aligned_text_tokens) as predict:
        for token in (torch.tensor([7, 10]), torch.tensor([8, 11]), torch.tensor([9, 12]), None):
            state, _, _ = model.streaming_step(state, token)
    assert predict.call_count == 2
    assert all(not call.kwargs.get('return_logits', False) for call in predict.call_args_list)
    assert state.alignment_debug_history is None


def test_turn_with_audio_but_no_registered_bpe_still_advances_timestamp_turn_index():
    model = _timestamp_model(confirmation_frames=1)
    state = _init_stream(model)
    for _ in range(3):
        state, _, _ = model.streaming_step(state, torch.full((2,), model.pad_id))
    assert state.timestamp_turns[0][0]['source_token_ids'] == []
    assert state.timestamp_turns[0][0]['has_audio']
    model.reset_attention_alignment_turn(state)
    state, _, _ = model.streaming_step(state, torch.tensor([7, 10]))
    traces = model.get_timestamps(state, finalize=True)
    for trace in traces:
        assert [turn['turn_index'] for turn in trace['turns']] == [0, 1]
        assert trace['timestamps'][0]['turn_index'] == 1


def test_unsupported_source_token_has_null_timing_instead_of_borrowing_the_next_token_boundary():
    model = _timestamp_model(confirmation_frames=1)
    state = _init_stream(model)
    with _attention_peaks([0, 0, 2, 2]):
        for token in (torch.tensor([7, 10]), torch.tensor([8, 11]), torch.tensor([9, 12]), None, None, None):
            state, _, _ = model.streaming_step(state, token)
    for trace in model.get_timestamps(state, finalize=True):
        assert trace['timestamps'][1]['start'] is None and trace['timestamps'][1]['end'] is None
        assert trace['timestamps'][2]['start'] is not None
        assert trace['timestamps'][2]['token_index'] == 2


def test_monotonic_timestamp_estimates_are_emitted_during_the_same_audio_step():
    settings = {
        'enabled': False,
        'record_timestamps': True,
        'timestamp_method': 'monotonic',
        'timestamp_confirmation_frames': 1,
    }
    model = _head_model(settings, {'frame_stacking_factor': 2, 'use_aligned_text_head': False})
    state = _init_stream(model)
    distributions = [[0.9, 0.09, 0.01], [0.2, 0.001, 0.799], [0.01, 0.001, 0.989]]
    assert all(max(range(3), key=lambda index: row[index]) != 1 for row in distributions)
    streamed = []
    with _attention_peaks(distributions):
        for token in (torch.tensor([7, 10]), torch.tensor([8, 11]), torch.tensor([9, 12]), None, None):
            state, _, _ = model.streaming_step(state, token)
            events = model.drain_timestamp_events(state)
            if events:
                traces = model.get_timestamps(state)
                assert all(
                    event['emitted_at'] <= traces[event['batch_index']]['audio_duration_seconds'] for event in events
                )
                streamed.extend(events)
    assert not state.finished.any()
    forced = [event for event in streamed if event['forced']]
    assert forced and all(event['confidence'] == 0.0 for event in forced)
    traces = model.get_timestamps(state, finalize=True)
    step = traces[0]['frame_step_seconds']
    for trace in traces:
        middle = trace['timestamps'][1]
        assert middle['token_index'] == 1 and middle['start'] is not None
        assert middle['end'] > middle['start']
        assert trace['method'] == 'monotonic' and not trace['guidance_enabled']
        assert trace['causal'] and 'lookahead_frames' not in trace
        assert middle['forced'] and middle['confidence'] == 0.0
        assert middle['start'] == pytest.approx(step)
        assert middle['end'] <= 2 * step
    assert any(event['token_index'] == 1 and event['event'] == 'start' for event in streamed)
    assert state.attention_alignment_inference_state.anchors == [0, 0]


def _timestamp_model(confirmation_frames):
    return _head_model(
        {
            'enabled': False,
            'record_timestamps': True,
            'timestamp_confirmation_frames': confirmation_frames,
            'timestamp_method': 'argmax',
        },
        {'frame_stacking_factor': 2, 'use_aligned_text_head': False},
    )


@contextmanager
def _attention_peaks(peaks):
    original = nemotron_h_decoder.apply_attention_alignment_inference
    iterator = iter(peaks)

    def capture(query, key, value, ordinary_output, request, **kwargs):
        result = original(query, key, value, ordinary_output, request, **kwargs)
        peak = next(iterator)
        for row in request.batch_indices:
            indices = request.state.last_raw_attention[row]['token_indices']
            probabilities = (
                [peak[index] for index in indices]
                if isinstance(peak, (list, tuple))
                else [0.9 if index == peak else 0.0 for index in indices]
            )
            request.state.last_raw_attention[row] = {'token_indices': indices, 'probabilities': probabilities}
        return result

    with patch.object(nemotron_h_decoder, 'apply_attention_alignment_inference', side_effect=capture):
        yield
