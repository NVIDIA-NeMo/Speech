# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from nemo.collections.tts.models.nemotron_tts.v2.base import NemotronTTSBase, TrainingMode
from nemo.collections.tts.modules.nemotron_h_decoder import NemotronHConfig, NemotronHModel
from nemo.collections.tts.modules.nemotron_tts.v2.acoustic_codes_predictor import AcousticCodesPredictor
from nemo.collections.tts.modules.nemotron_tts.v2.timestamps import StreamingTimestampSession, resolve_timestamp_config

pytestmark = pytest.mark.unit


def test_timestamp_recording_defaults_off_and_can_be_enabled_without_training_modules():
    model = _model()
    state = _init_stream(model)
    assert state.timestamp_session is None
    assert model.get_timestamps(state) is None and model.drain_timestamp_events(state) == []
    assert model.drain_timestamp_frames(state) == []
    assert not hasattr(model, 'text_aligner') and not hasattr(model, 'aligned_text_head')
    model.configure_timestamps(enabled=True)
    assert _init_stream(model).timestamp_session is not None


@pytest.mark.parametrize('use_cfg', [False, True])
def test_timestamp_recording_preserves_real_cached_backbone_outputs_and_audio_codes(use_cfg):
    baseline, observed = _model(), _model(enabled=True)
    states = [_init_stream(model, use_cfg=use_cfg, ragged=True) for model in (baseline, observed)]
    tokens = [torch.tensor([7, 10]), torch.tensor([8, 11]), torch.tensor([9, 12])]
    tokens += [torch.full((2,), baseline.eos_id)] * 3 + [None] * 4
    with patch(
        'nemo.collections.tts.modules.glow_tts_aligner.TinyGlowTTSAligner.forward',
        side_effect=AssertionError('The acoustic aligner must never run during timestamp inference'),
    ):
        for token in tokens:
            result = [model.streaming_step(state, token) for model, state in zip((baseline, observed), states)]
            states = [value[0] for value in result]
            torch.testing.assert_close(states[0].last_hidden, states[1].last_hidden, rtol=0, atol=0)
            if result[0][1] is not None:
                torch.testing.assert_close(result[0][1], result[1][1], rtol=0, atol=0)
    trace = observed.get_timestamps(states[1])
    assert trace[0]['causal'] and not trace[0]['guidance_enabled']
    assert all(
        token['token_id'] not in (observed.bos_id, observed.eos_id, observed.pad_id)
        for item in trace
        for token in item['timestamps']
    )
    assert states[1].last_hidden.size(0) == (4 if use_cfg else 2)
    frames = observed.drain_timestamp_frames(states[1])
    assert frames and all(frame['batch_index'] < 2 for frame in frames)
    assert len(frames) == sum(item['frame_count'] for item in trace)
    assert observed.drain_timestamp_frames(states[1]) == []


def test_bos_delay_five_and_stacked_frames_have_no_timestamp_index_shift():
    model = _model(enabled=True)
    state = _init_stream(model, use_cfg=True)
    with _attention_peaks([0, 0, 1, 2, 2]):
        for token in [model.bos_id, 7, 7, 8, model.eos_id]:
            state, codes, _ = model.streaming_step(state, torch.full((2,), token))
            assert codes is None and model.drain_timestamp_events(state) == []
            assert model.drain_timestamp_frames(state) == []
        state, codes, _ = model.streaming_step(state, None)
        events = model.drain_timestamp_events(state, [0])
        assert codes is not None and events
        assert events[0]['event'] == 'start' and events[0]['token_id'] == 7
        assert events[0]['start'] == 0.0 and events[0]['end'] == pytest.approx(0.08)
        assert events[0]['emitted_at'] == pytest.approx(0.08)
        assert all(event['batch_index'] == 0 for event in events)
        assert all(event['batch_index'] == 1 for event in model.drain_timestamp_events(state))
        frames = model.drain_timestamp_frames(state, [0])
        assert len(frames) == 1 and frames[0]['token_id'] == 7
        assert frames[0]['codec_frame_start'] == 0 and frames[0]['codec_frame_end'] == 2
        assert frames[0]['frame_index'] == 0 and frames[0]['end'] == pytest.approx(0.08)
        assert all(frame['batch_index'] == 1 for frame in model.drain_timestamp_frames(state))
        assert not state.finished.any()
        for _ in range(4):
            state, _, _ = model.streaming_step(state, None)
            assert model.drain_timestamp_events(state)
            frames = model.drain_timestamp_frames(state)
            assert len(frames) == 2 and {frame['batch_index'] for frame in frames} == {0, 1}
    for trace in model.get_timestamps(state, finalize=True):
        assert [record['token_id'] for record in trace['timestamps']] == [7, 7, 8]
        assert [record['token_index'] for record in trace['timestamps']] == [0, 1, 2]
        assert [boundary for record in trace['timestamps'] for boundary in (record['start'], record['end'])] == (
            pytest.approx([0.0, 0.16, 0.16, 0.24, 0.24, 0.40])
        )
        assert trace['frame_count'] == len(trace['frames']) == 5
        assert [frame['token_index'] for frame in trace['frames']] == [0, 0, 1, 2, 2]
        assert [frame['codec_frame_start'] for frame in trace['frames']] == [0, 2, 4, 6, 8]


@pytest.mark.parametrize('eos_subframe', [0, 1])
def test_audio_eos_clips_partial_stacked_frame_before_timestamp_events(eos_subframe):
    model = _model(enabled=True)
    state = _init_stream(model)

    def predict(current, needs_audio=None):
        codes = torch.zeros(2, model.num_audio_codebooks, 2, dtype=torch.long)
        codes[:, :, eos_subframe] = model.audio_eos_id
        codes = codes.reshape(2, -1)
        return codes, codes

    model._predict_audio_codes = predict
    with _attention_peaks([0]):
        for token in [model.bos_id, 7, model.eos_id, model.pad_id, model.pad_id]:
            state, _, _ = model.streaming_step(state, torch.full((2,), token))
        state, _, _ = model.streaming_step(state, None)
    assert state.finished.all()
    output = model.streaming_finalize(state)
    for trace in output.timestamps:
        token = trace['timestamps'][0]
        assert token['token_id'] == 7
        assert token['end'] == (pytest.approx(0.04) if eos_subframe else None)
        assert trace['frame_count'] == len(trace['frames']) == eos_subframe
        if eos_subframe:
            frame = trace['frames'][0]
            assert frame['codec_frame_start'] == 0 and frame['codec_frame_end'] == 1
            assert frame['valid_codec_frames'] == 1 and frame['end'] == pytest.approx(0.04)
    assert all(event['end'] <= eos_subframe * 0.04 + 1e-8 for event in model.drain_timestamp_events(state))


@pytest.mark.parametrize('profile', ['bulk', 'step'])
def test_multiturn_user_profile_keys_are_not_registered_as_response_text(profile):
    model = _model(enabled=True)
    state = _init_stream(model)
    for token in [7, 8, 9, 10, 11, 12]:
        state, _, _ = model.streaming_step(state, torch.full((2,), token))
    before = model.get_timestamps(state)[0]
    assert before['timestamps']
    if profile == 'bulk':
        state = model.streaming_prefill(state, torch.tensor([[20, 21], [22, 23]]))
    else:
        state, _, _ = model.streaming_step(
            state, torch.tensor([20, 22]), prefill_like_step=True, prefill_like_is_last_step=True
        )
    with torch.inference_mode():
        state.turn_text_tokens_seen.zero_()
        state.text_finished.zero_()
        state.audio_steps.zero_()
    state, _, _ = model.streaming_step(state, torch.full((2,), 13))
    trace = model.get_timestamps(state)[0]
    current = trace['turns'][-1]
    assert current['turn_index'] > before['turns'][-1]['turn_index']
    assert current['source_token_ids'] == [13]
    assert all(record['token_id'] not in [20, 21] for record in trace['timestamps'])


def test_text_after_eos_is_not_mistaken_for_conditioning_input():
    model = _model(enabled=True)
    state = _init_stream(model)
    for token in [7, model.eos_id, 8, 9]:
        state, _, _ = model.streaming_step(state, torch.full((2,), token))
    assert model.get_timestamps(state)[0]['turns'][0]['source_token_ids'] == [7]


def test_dropout_audio_has_an_unaligned_frame_and_cannot_reuse_warmup_attention():
    model = _model(enabled=True)
    state = _init_stream(model)
    with _attention_peaks([0]):
        for token in [7, 8, 9, model.pad_id, model.pad_id]:
            state, codes, _ = model.streaming_step(
                state,
                torch.full((2,), token),
                prefill_like_step=True,
                timestamp_text_warmup=True,
            )
            assert codes is None and model.drain_timestamp_frames(state) == []
        assert state.timestamp_session._raw_attention is not None
        state, codes, _ = model.streaming_step(state, None, force_dropout_text=True)
        assert codes is not None
        missing = model.drain_timestamp_frames(state)
        assert len(missing) == 2
        assert all(frame['status'] == 'unaligned' and frame['token_index'] is None for frame in missing)
        assert all(frame['confidence'] is None and frame['frame_index'] == 0 for frame in missing)
        assert all(frame['codec_frame_start'] == 0 and frame['codec_frame_end'] == 2 for frame in missing)
        state, codes, _ = model.streaming_step(state, None)
        recovered = model.drain_timestamp_frames(state)
        assert codes is not None and len(recovered) == 2
        assert all(frame['token_id'] == 7 and frame['frame_index'] == 1 for frame in recovered)
        assert all(frame['codec_frame_start'] == 2 and frame['codec_frame_end'] == 4 for frame in recovered)
        assert all(frame['start'] == pytest.approx(0.08) for frame in recovered)
    assert all(trace['frame_count'] == 2 for trace in model.get_timestamps(state))


def _model(enabled=None):
    """Run the actual V2 streaming/decoder code with only codec/context I/O stubbed."""
    torch.manual_seed(5)
    model = NemotronTTSBase.__new__(NemotronTTSBase)
    nn.Module.__init__(model)
    model._cfg = OmegaConf.create(
        dict(
            embedding_dim=16,
            hidden_dim=16,
            use_multiturn_dataset=True,
            use_user_speaking_token=True,
            use_user_speaking_end_token=True,
        )
    )
    model.timestamp_config = resolve_timestamp_config(None if enabled is None else {'enabled': enabled})
    model.num_audio_codebooks = 2
    model.frame_stacking_factor = 2
    model.sample_rate = model.output_sample_rate = 100
    model.codec_model_samples_per_frame = 4
    model.codebook_size = 16
    model.audio_bos_id, model.audio_eos_id = 16, 17
    model.mask_token_id, model.audio_user_speaking_id, model.audio_user_speaking_end_id = 20, 21, 22
    model.bos_id, model.eos_id, model.cfg_unk_token_id, model.pad_id = 60, 61, 62, 0
    model.phoneme_tokenizer = None
    model.tokenizer = SimpleNamespace(pad=0)
    model.disable_subword_embedding = model.use_bpe_char_tokenizer = False
    config = NemotronHConfig(
        hidden_size=16,
        num_hidden_layers=5,
        hybrid_override_pattern='*-*-*',
        vocab_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=32,
        mamba_num_heads=2,
        mamba_head_dim=8,
        ssm_state_size=8,
        n_groups=1,
        _attn_implementation='sdpa',
    )
    model.decoder = NemotronHModel(config)
    model.decoder.set_input_embeddings(nn.Embedding(64, 16))
    model.audio_test_embedding = nn.Embedding(32, 16)

    def embed(codes, codebook_indices=None):
        return model.audio_test_embedding(codes.long()).mean(dim=1)

    model.embed_audio_tokens = embed
    model.acoustic_codes_predictor = AcousticCodesPredictor(
        backbone_config=config,
        embed_codes=embed,
        num_audio_codebooks=4,
        audio_eos_id=model.audio_eos_id,
        mask_token_id=model.mask_token_id,
        codebook_size=model.codebook_size,
        prediction_schedule=[2, 2],
        n_layers=1,
    )
    model.register_buffer('_codec_sil_codes_buffer', torch.zeros(2, dtype=torch.long), persistent=False)
    mode = TrainingMode('streaming', 3, 5, 0)
    model.mode_name_to_mode = {mode.name: mode}
    model.default_inference_mode = mode.name

    def context(**kwargs):
        text = kwargs['context_text_tokens']
        return (
            model.embed_text_tokens(text),
            kwargs['context_text_tokens_lens'],
            kwargs['context_audio_codes'],
            kwargs['context_audio_codes_lens'],
        )

    model.prepare_context_tensors = context

    def predict(state, needs_audio=None):
        codes = torch.zeros(state.config.batch_size, 4, dtype=torch.long, device=state.config.device)
        return codes, codes

    model._predict_audio_codes = predict

    def decode(codes, lengths):
        samples = lengths * model.codec_model_samples_per_frame
        return torch.zeros(codes.size(0), int(samples.max()), device=codes.device), samples, codes

    model._codec_helper = SimpleNamespace(codes_to_audio=decode)
    return model.eval()


def _init_stream(model, use_cfg=False, ragged=False):
    return model.streaming_init(
        context_text_tokens=torch.tensor([[3, 4], [5, 6]]),
        context_text_tokens_lens=torch.tensor([2, 1 if ragged else 2]),
        context_audio_codes=torch.zeros(2, 2, 2, dtype=torch.long),
        context_audio_codes_lens=torch.tensor([2, 2]),
        use_cfg=use_cfg,
        cfg_scale=2.5 if use_cfg else 1.0,
    )


@contextmanager
def _attention_peaks(indices):
    original = StreamingTimestampSession.observe_attention

    def observe(session, query, key, **kwargs):
        original(session, query, key, **kwargs)
        posterior = torch.zeros(session.batch_size, key.size(-2))
        for row in range(session.batch_size):
            origin, end = session._origins[row], session._ends[row]
            step = 0 if origin is None else (end - origin) // session.frame_stacking_factor
            turn = session._turns[row][-1]
            index = indices[min(step, len(indices) - 1)]
            if index < len(turn['keys']):
                posterior[row, turn['keys'][index]] = 1.0
        session._raw_attention = posterior

    with patch.object(StreamingTimestampSession, 'observe_attention', observe):
        yield
