# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from nemo.collections.tts.models.nemotron_tts.v2.model import NemotronTTSModel
from nemo.collections.tts.modules.nemotron_tts.v2.prosody import ProsodyChannel
from tests.collections.tts.models.test_easy_magpietts import (
    _codec_restore_uses_random_initialized_audio_codec,
    _seed_everything,
    _toy_batch,
    _toy_codes,
    tiny_easy_magpie_cfg,
)

pytestmark = pytest.mark.unit


def _channel(**kwargs):
    return ProsodyChannel(embedding_dim=32, hidden_dim=32, input_sample_rate=24000, encoder_channels=16, **kwargs)


def _model(stacking=1, enabled=True):
    cfg = tiny_easy_magpie_cfg(
        {
            'frame_stacking_factor': stacking,
            'use_codec_latent_audio_embedding': True,
            'acoustic_codes_predictor_schedule': [8 * stacking],
            'acoustic_codes_predictor_n_layers': 1,
            'prosody': {'enabled': enabled, 'encoder_channels': 16},
            'training_modes': [
                {'text_input_mode': 'streaming', 'streaming_phonemes_delay': 0, 'streaming_speech_delay': 5}
            ],
        }
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _seed_everything()
    with _codec_restore_uses_random_initialized_audio_codec():
        model = NemotronTTSModel(cfg)
    return model.eval().to(device)


def _batch(model):
    device = next(model.parameters()).device
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in _toy_batch(model).items()
    }
    batch['audio_codes'] = _toy_codes(model, 2, 32).to(device)
    batch['audio_codes_lens'] = torch.tensor([32, 28], device=device)
    # Two synthetic waveforms, 0.64 and 0.56 seconds, at the tiny codec's 24 kHz.
    time = torch.arange(15360, device=device) / model.sample_rate
    batch['audio'] = torch.stack([0.2 * (time * hz * 2 * torch.pi).sin() for hz in (180, 240)])
    batch['audio_lens'] = torch.tensor([15360, 13440], device=device)
    batch.pop('agent_mask')
    batch.pop('task')
    return batch


def test_prosody_encoder_rate_partial_frames_and_embedding_round_trip():
    channel = _channel()
    audio = torch.randn(2, 24000) * 0.1
    quantized, codes, lengths = channel.encode(audio, torch.tensor([24000, 100]))
    assert lengths.tolist() == [7, 1]  # ceil(duration * 6.25)
    assert codes.shape == (2, 7)
    assert codes.min() >= 0 and codes.max() < 2048
    assert channel.quantizer.num_codebooks == 1
    torch.testing.assert_close(channel.embed(codes, quantized=quantized)[0], channel.embed(codes)[0])
    # Every ID is reachable and stable through decode/encode.
    all_codes = torch.arange(2048)[None]
    decoded = channel.quantizer.decode(indices=all_codes[None].int())
    assert torch.equal(channel.quantizer.codes_to_indices(decoded).long(), all_codes)


def test_prosody_timing_dropout_and_padding_loss():
    channel = _channel(dropout=1.0)
    codes = torch.tensor([[3, 7, 9], [2, 8, 0]])
    quantized = channel.quantizer.decode(indices=codes[None].int())
    embedded, targets, lengths = channel.teacher_force(
        quantized, codes, torch.tensor([3, 2]), torch.tensor([10, 8]), speech_delay=5
    )
    pad = channel.pad_id
    assert lengths.tolist() == [15, 13]
    assert targets[0].tolist() == [pad] * 4 + [3, 7, 9] + [pad] * 8
    assert targets[1].tolist() == [pad] * 4 + [2, 8] + [pad] * 9
    inputs = F.pad(targets[:, :-1], (1, 0), value=pad)
    torch.testing.assert_close(embedded, channel.embed(inputs))
    dropped = channel.embed(inputs, apply_dropout=True)
    torch.testing.assert_close(dropped[inputs < pad], channel.special_embeddings.weight[1].expand(5, -1))
    torch.testing.assert_close(dropped[inputs == pad], embedded[inputs == pad])

    hidden = torch.randn(2, 15, 32, requires_grad=True)
    loss, _ = channel.compute_loss(hidden, targets, lengths)
    loss.backward()
    assert hidden.grad[0, -1].abs().sum() > 0  # real PAD tail is supervised
    assert hidden.grad[1, 12].abs().sum() > 0
    assert hidden.grad[1, 13:].abs().sum() == 0  # unrelated batch padding is excluded


@pytest.mark.parametrize('stacking', [1, 2])
@pytest.mark.parametrize('mixed_precision', [False, True])
def test_prosody_process_batch_and_encoder_gradients(stacking, mixed_precision):
    model = _model(stacking)
    model.prosody.dropout = 0
    with torch.autocast(
        device_type=next(model.parameters()).device.type, dtype=torch.bfloat16, enabled=mixed_precision
    ):
        output = model.process_batch(**_batch(model), mode='train')
    assert torch.isfinite(output.loss)
    torch.testing.assert_close(
        output.loss, output.codebook_loss + output.acoustic_codes_predictor_loss + output.prosody_loss
    )
    assert output.prosody_codes_target[0, :4].eq(model.prosody.pad_id).all()
    assert output.prosody_codes_target[0, 4:8].lt(2048).all()
    assert output.prosody_codes_target[0, 8:].eq(model.prosody.pad_id).all()
    assert output.prosody_codes_lens_target.tolist() == (output.audio_codes_lens_target + 5).tolist()
    # Acoustic reconstruction alone must train the encoder, through FSQ and its input projection.
    output.codebook_loss.backward()
    gradients = [parameter.grad for parameter in model.prosody.encoder.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum() for gradient in gradients) > 0


def test_prosody_requires_waveform_and_valid_delay():
    model = _model()
    batch = _batch(model)
    batch.pop('audio')
    with pytest.raises(ValueError, match='requires target'):
        model.process_batch(**batch)
    with pytest.raises(ValueError, match='streaming_speech_delay'):
        with _codec_restore_uses_random_initialized_audio_codec():
            NemotronTTSModel(
                tiny_easy_magpie_cfg(
                    {
                        'use_codec_latent_audio_embedding': True,
                        'prosody': {'enabled': True, 'lead_steps': 3},
                    }
                )
            )


def test_prosody_disabled_preserves_v2_losses_and_state():
    model = _model(enabled=False)
    output = model.process_batch(**_batch(model))
    torch.testing.assert_close(output.loss, output.codebook_loss + output.acoustic_codes_predictor_loss)
    assert output.prosody_loss is None
    assert output.prosody_logits is None
    assert model.prosody is None
    assert not any(key.startswith('prosody.') for key in model.state_dict())


@pytest.mark.parametrize('use_cfg', [False, True])
def test_prosody_streaming_delay_feedback_padding_and_batch_context(use_cfg):
    model = _model()
    batch = _batch(model)
    channel = model.prosody
    with torch.no_grad():
        # Predict code 17 whenever real prosody is enabled, then switch to PAD.
        channel.output_projection.weight.zero_()
        channel.output_projection.bias.fill_(-100)
        channel.output_projection.bias[17] = 100
    state = model.streaming_init(
        **{
            key: batch[key]
            for key in (
                'context_audio_codes',
                'context_audio_codes_lens',
                'context_text_tokens',
                'context_text_tokens_lens',
            )
        },
        use_cfg=use_cfg,
        temperature=0,
    )
    first_real_seen = torch.zeros(2, dtype=torch.bool, device=state.config.device)
    # Avoid random acoustic EOS so that the PAD feedback can be observed.
    audio_codes = torch.zeros(2, model.num_audio_codebooks, dtype=torch.long, device=state.config.device)
    with patch.object(model, '_predict_audio_codes', return_value=(audio_codes, audio_codes)):
        for _ in range(12):
            step = state.turn_text_tokens_seen.clone()
            needs_context = state.context_position < state.full_context_lens
            previous = state.last_prosody_codes
            if previous is None:
                previous = torch.full_like(step, channel.pad_id)
            with patch.object(channel, 'embed', wraps=channel.embed) as embed:
                state, _, _ = model.streaming_step(state, text_tokens=None)
                observed = embed.call_args.args[0][:, 0]
            expected = torch.where(step <= 4, channel.pad_id, previous)
            active = ~needs_context
            torch.testing.assert_close(observed[active], expected[active])
            if use_cfg:
                assert state.last_hidden.size(0) == 4
            first_real_seen |= active & (observed == 17)
            if first_real_seen.all():
                with torch.no_grad():
                    channel.output_projection.bias[17] = -100
                    channel.output_projection.bias[channel.pad_id] = 100
    assert first_real_seen.all()
    assert state.last_prosody_codes.eq(channel.pad_id).all()


@torch.inference_mode()
def test_prosody_training_and_streaming_embeddings_match():
    model = _model(stacking=2)
    batch = _batch(model)
    with patch.object(model, 'forward', wraps=model.forward) as forward:
        output = model.process_batch(**batch, mode='val')
        training_embeddings = forward.call_args.kwargs['inputs_embeds']
    state = model.streaming_init(
        **{
            key: batch[key]
            for key in (
                'context_audio_codes',
                'context_audio_codes_lens',
                'context_text_tokens',
                'context_text_tokens_lens',
            )
        },
        temperature=0,
    )
    zero_delay = torch.zeros_like(batch['audio_codes_lens'])
    state.gt_audio_embeddings, state.gt_audio_lens, *_ = model.prepare_audio_channel_embeddings(
        batch['audio_codes'], batch['audio_codes_lens'], delay=zero_delay
    )
    indices = torch.arange(2, device=state.config.device)
    for position in range(state.cache_seq_len, training_embeddings.size(1)):
        step = state.turn_text_tokens_seen
        state.audio_steps = (step - 5).clamp(min=0)
        state.finished = step >= output.prosody_codes_lens_target
        previous = output.prosody_codes_target[
            indices, (step - 1).clamp(min=0, max=output.prosody_codes_target.size(1) - 1)
        ]
        state.last_prosody_codes = torch.where(step == 0, model.prosody.pad_id, previous)
        text = batch['text'][indices, state.text_tokens_seen.clamp(max=batch['text'].size(1) - 1)]
        text = torch.where(state.text_tokens_seen >= batch['text_lens'], model.eos_id, text)
        embedded, needs_context, _, _ = model._prepare_streaming_input(state, text, force_dropout_text=False)
        valid = position < state.context_lens + output.prosody_codes_lens_target
        torch.testing.assert_close(embedded[valid, 0], training_embeddings[valid, position], atol=1e-5, rtol=1e-5)
        state.context_position += needs_context.long()
        state.turn_text_tokens_seen += (~needs_context).long()
        state.text_tokens_seen += (~needs_context).long()


def test_prosody_toy_training_five_steps():
    model = _model()
    model.train()
    batch = _batch(model)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    before = model.prosody.encoder.post_conv.conv.weight.detach().clone()
    losses = []
    logged = {}
    with patch.object(
        model, 'log', side_effect=lambda name, value, **kwargs: logged.update({name: torch.as_tensor(value).detach()})
    ):
        for step in range(5):
            optimizer.zero_grad(set_to_none=True)
            loss = model.training_step(batch, step)
            loss.backward()
            parameters = [p for p in model.parameters() if p.grad is not None]
            assert all(torch.isfinite(p.grad).all() for p in parameters)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            print(f'prosody toy step={step + 1} total={losses[-1]:.4f} prosody={logged["train/prosody_loss"]:.4f}')
    assert torch.isfinite(torch.tensor(losses)).all()
    assert not torch.equal(before, model.prosody.encoder.post_conv.conv.weight)
    assert losses[-1] < losses[0]
