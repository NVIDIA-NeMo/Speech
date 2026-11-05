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
    _padded_token_tensor,
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
            'prosody': {'enabled': enabled, 'encoder_channels': 16, 'latent_dim': 16, 'alignment_dim': 8},
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
    text = torch.randn(2, 5, 32)
    output = channel.encode(audio, torch.tensor([24000, 100]), text, torch.tensor([5, 2]))
    assert output.audio_lens.tolist() == [4, 1]  # ceil(duration * 3.125)
    assert output.code_lens.tolist() == [5, 2]
    assert output.codes.shape == (2, 5)
    assert output.quantized.shape == (2, 4, 5)  # quantization happens after alignment
    assert output.codes[0].min() >= 0 and output.codes[0].max() < 2048
    assert output.codes[1, 2:].eq(channel.pad_id).all()
    assert channel.quantizer.num_codebooks == 1
    torch.testing.assert_close(
        channel.embed(output.codes, quantized=output.quantized)[0], channel.embed(output.codes)[0]
    )
    # Every ID is reachable and stable through decode/encode.
    all_codes = torch.arange(2048)[None]
    decoded = channel.quantizer.decode(indices=all_codes[None].int())
    assert torch.equal(channel.quantizer.codes_to_indices(decoded).long(), all_codes)


def test_mas_pools_audio_or_reuses_coarse_latents_and_detaches_text():
    channel = _channel(latent_dim=4, alignment_dim=4, alignment_temperature=1)
    with torch.no_grad():
        channel.audio_alignment_projection.weight.copy_(torch.eye(4))
        channel.audio_alignment_projection.bias.zero_()
        channel.text_alignment_projection.weight.zero_()
        channel.text_alignment_projection.weight[:, :4].copy_(torch.eye(4))
        channel.text_alignment_projection.bias.zero_()
    latents = torch.zeros(2, 4, 5)
    latents[0, 0, :2] = torch.tensor([2, 4])
    latents[0, 1, 2:] = torch.tensor([1, 3, 5])
    latents[1, 0, 0] = 2
    latents[1, 1, 1] = 4
    latents.requires_grad_()
    text = torch.zeros(2, 5, 32)
    text[0, 0, 0] = text[0, 1, 1] = 1
    text[1, :3, 0] = text[1, 3:, 1] = 1
    text.requires_grad_()
    pooled, path, loss = channel.align_and_pool(latents, torch.tensor([5, 2]), text, torch.tensor([2, 5]))
    torch.testing.assert_close(path[0, :2], torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 1.0]]))
    assert path[0, 2:].sum() == 0
    torch.testing.assert_close(path[1, :, :2], torch.tensor([[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 2))
    assert path[1, :, 2:].sum() == 0
    torch.testing.assert_close(pooled[0, :, 0], torch.tensor([3.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(pooled[0, :, 1], torch.tensor([0.0, 3.0, 0.0, 0.0]))
    torch.testing.assert_close(pooled[1, 0, :3], torch.full((3,), 2.0))
    torch.testing.assert_close(pooled[1, 1, 3:], torch.full((2,), 4.0))
    assert torch.isfinite(loss)
    (loss + pooled.square().mean()).backward()
    assert text.grad is None
    assert latents.grad is not None and torch.isfinite(latents.grad).all()
    assert channel.text_alignment_projection.weight.grad.abs().sum() > 0
    assert channel.audio_alignment_projection.weight.grad.abs().sum() > 0


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


def test_prosody_never_truncates_text_codes_at_acoustic_eos():
    channel = _channel()
    codes = torch.arange(12)[None]
    quantized = channel.quantizer.decode(indices=codes[None].int())
    _, targets, lengths = channel.teacher_force(
        quantized, codes, torch.tensor([12]), torch.tensor([2]), speech_delay=5
    )
    assert lengths.tolist() == [17]
    assert targets[0].tolist() == [channel.pad_id] * 4 + list(range(12)) + [channel.pad_id]


def test_prosody_empty_text_has_no_codes_and_only_pad_supervision():
    channel = _channel()
    output = channel.encode(
        torch.randn(1, 24000) * 0.1, torch.tensor([24000]), torch.empty(1, 0, 32), torch.tensor([0])
    )
    assert output.codes.shape == (1, 0)
    assert output.code_lens.tolist() == [0]
    assert output.alignment_loss == 0
    _, targets, _ = channel.teacher_force(output.quantized, output.codes, output.code_lens, torch.tensor([10]), 5)
    assert targets.eq(channel.pad_id).all()


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
        output.loss,
        output.codebook_loss
        + output.acoustic_codes_predictor_loss
        + output.prosody_loss
        + model.prosody.alignment_loss_scale * output.prosody_alignment_loss,
    )
    batch = _batch(model)
    assert output.prosody_code_lens.tolist() == batch['text_lens'].tolist()
    for row, count in enumerate(batch['text_lens'].tolist()):
        assert output.prosody_codes_target[row, :4].eq(model.prosody.pad_id).all()
        assert output.prosody_codes_target[row, 4 : 4 + count].lt(2048).all()
        assert output.prosody_codes_target[row, 4 + count :].eq(model.prosody.pad_id).all()
    assert output.prosody_codes_lens_target.tolist() == (output.audio_codes_lens_target + 5).tolist()
    # Acoustic reconstruction alone must train the encoder, through FSQ and its input projection.
    output.codebook_loss.backward()
    gradients = [parameter.grad for parameter in model.prosody.encoder.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum() for gradient in gradients) > 0


def test_alignment_loss_does_not_change_shared_text_embeddings():
    model = _model()
    batch = _batch(model)
    batch['text'], batch['text_lens'] = _padded_token_tensor(model, ['Please read every word here.', 'Hello.'])
    batch['text'] = batch['text'].to(next(model.parameters()).device)
    batch['text_lens'] = batch['text_lens'].to(next(model.parameters()).device)
    batch['audio_codes'] = batch['audio_codes'][:, :, :2]
    batch['audio_codes_lens'] = torch.full_like(batch['audio_codes_lens'], 2)
    output = model.process_batch(**batch, mode='train')
    assert output.prosody_code_lens.tolist() == batch['text_lens'].tolist()
    assert (output.prosody_codes_target < 2048).sum(1).tolist() == batch['text_lens'].tolist()
    assert (output.prosody_codes_lens_target >= batch['text_lens'] + 5).all()
    output.prosody_alignment_loss.backward()
    assert all(p.grad is None for p in model.text_embedding.parameters())
    assert all(p.grad is None for p in model.cas_encoder.parameters())
    assert model.prosody.text_alignment_projection.weight.grad.abs().sum() > 0


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
        # Prefer PAD, but every valid text slot must still predict a real code.
        channel.output_projection.weight.zero_()
        channel.output_projection.bias.fill_(-100)
        channel.output_projection.bias[17] = 100
        channel.output_projection.bias[channel.pad_id] = 200
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
    real_predictions = torch.zeros(2, dtype=torch.long, device=state.config.device)
    # Avoid random acoustic EOS so that the PAD feedback can be observed.
    audio_codes = torch.zeros(2, model.num_audio_codebooks, dtype=torch.long, device=state.config.device)
    with patch.object(model, '_predict_audio_codes', return_value=(audio_codes, audio_codes)):
        for _ in range(16):
            step = state.turn_text_tokens_seen.clone()
            needs_context = state.context_position < state.full_context_lens
            positions = state.text_tokens_seen.clamp(max=batch['text'].size(1) - 1)
            text = batch['text'][torch.arange(2, device=state.config.device), positions]
            text = torch.where(state.text_tokens_seen >= batch['text_lens'], model.eos_id, text)
            with patch.object(channel, 'embed', wraps=channel.embed) as embed:
                state, _, _ = model.streaming_step(state, text_tokens=text)
                observed = embed.call_args.args[0][:, 0]
            expected = torch.where((step >= 5) & (step < 5 + batch['text_lens']), 17, channel.pad_id)
            active = ~needs_context
            torch.testing.assert_close(observed[active], expected[active])
            if use_cfg:
                assert state.last_hidden.size(0) == 4
            first_real_seen |= active & (observed == 17)
            real_predictions += (active & (state.last_prosody_codes < channel.pad_id)).long()
    assert first_real_seen.all()
    assert real_predictions.tolist() == batch['text_lens'].tolist()
    assert state.prosody_text_lens.tolist() == batch['text_lens'].tolist()
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
