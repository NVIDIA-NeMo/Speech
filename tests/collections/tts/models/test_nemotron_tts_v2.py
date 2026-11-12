# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from nemo.collections.tts.models.nemotron_tts.v2.model import NemotronTTSModel
from tests.collections.tts.models.test_easy_magpietts import (
    _codec_restore_uses_random_initialized_audio_codec,
    _seed_everything,
    _toy_batch,
    tiny_easy_magpie_cfg,
)

pytestmark = pytest.mark.unit


@pytest.fixture(params=[None, True, False], ids=['default', 'enabled', 'disabled'])
def model(request):
    return _make_model(request.param)


@pytest.mark.parametrize('mode', ['train', 'val'])
def test_auxiliary_loss_controls_projection_and_gradients(model, mode):
    enabled = model.cfg.get('compute_auxiliary_codebook_loss', True)
    with (
        patch.object(model.audio_out_projection, 'forward', wraps=model.audio_out_projection.forward) as projection,
        patch.object(model.final_proj, 'forward', wraps=model.final_proj.forward) as head,
        patch.object(model, 'compute_loss', wraps=model.compute_loss) as compute_loss,
    ):
        output = model.process_batch(**_toy_batch(model), mode=mode)

    assert model.compute_auxiliary_codebook_loss == enabled
    assert projection.call_count == head.call_count == compute_loss.call_count == int(enabled)
    expected_loss = model.acoustic_codes_predictor_loss_scale * output.acoustic_codes_predictor_loss
    if enabled:
        assert torch.isfinite(output.codebook_loss)
        assert output.logits.shape[:2] == output.audio_codes_target.shape[0::2]
        expected_loss = expected_loss + model.parallel_codebook_loss_scale * output.codebook_loss
    else:
        assert output.codebook_loss is None
        assert output.logits is None
    torch.testing.assert_close(output.loss, expected_loss)
    assert torch.isfinite(output.loss) and output.loss.item() > 0

    output.loss.backward()
    for module in (model.audio_out_projection, model.final_proj):
        for param in module.parameters():
            assert param.requires_grad == enabled
            assert (param.grad is not None) == enabled
    assert any(p.grad is not None and p.grad.count_nonzero() > 0 for p in model.decoder.parameters())
    assert any(p.grad is not None and p.grad.count_nonzero() > 0 for p in model.acoustic_codes_predictor.parameters())


def test_auxiliary_loss_logging_and_validation_audio(model):
    enabled = model.compute_auxiliary_codebook_loss
    model._trainer = SimpleNamespace(global_rank=0, local_rank=0, world_size=1, loggers=[])
    with (
        patch.object(model, 'log') as log,
        patch.object(model, 'log_dict'),
        patch.object(model, 'log_val_audio_example', return_value={}) as log_audio,
    ):
        loss = model.training_step(_toy_batch(model), batch_idx=0)
        val_output = model.validation_step(_toy_batch(model), batch_idx=0)
        model.on_validation_epoch_end()

    assert torch.isfinite(loss)
    assert ('val_codebook_loss' in val_output) == enabled
    assert log_audio.call_count == int(enabled)
    logged_names = {call.args[0] for call in log.call_args_list}
    assert ('train/codebook_loss' in logged_names) == enabled
    assert ('val/codebook_loss' in logged_names) == enabled
    assert {'train/acoustic_codes_predictor_loss', 'val/acoustic_codes_predictor_loss'} <= logged_names
    assert not model.validation_step_outputs


@pytest.mark.parametrize('drop_phonemes', [False, True])
def test_phoneme_loss_without_auxiliary_logits(drop_phonemes):
    model = _make_model(False)
    model.phoneme_tokenizer = SimpleNamespace(bos_token_id=1, eos_token_id=2, pad=0)
    model.phoneme_vocab_size = 32
    model.phoneme_stacking_factor = 1
    model.phoneme_embedding = nn.Embedding(32, 32)
    model.phoneme_final_proj = nn.Linear(32, 32)
    model.embed_phoneme_tokens = lambda tokens: model.phoneme_embedding(tokens.long()).sum(dim=1)
    batch = _toy_batch(model)
    batch['phoneme_tokens'] = torch.tensor([[3, 4, 5], [6, 7, 8]])
    batch['phoneme_tokens_lens'] = torch.tensor([3, 3])
    batch['phoneme_turn_dropout'] = torch.tensor([drop_phonemes, drop_phonemes])

    output = model.process_batch(**batch, mode='val')

    assert output.logits is None
    assert (output.phoneme_loss.item() == 0) == drop_phonemes
    torch.testing.assert_close(
        output.loss,
        model.acoustic_codes_predictor_loss_scale * output.acoustic_codes_predictor_loss
        + model.phoneme_loss_weight * output.phoneme_loss,
    )
    output.loss.backward()
    if not drop_phonemes:
        assert model.phoneme_final_proj.weight.grad.count_nonzero() > 0


def test_disabling_auxiliary_loss_preserves_checkpoint_keys():
    enabled = _make_model(True)
    disabled = _make_model(False)

    assert enabled.state_dict().keys() == disabled.state_dict().keys()
    disabled.load_state_dict(enabled.state_dict(), strict=True)
    enabled.load_state_dict(disabled.state_dict(), strict=True)


def _make_model(compute_auxiliary_codebook_loss):
    _seed_everything()
    overrides = {
        'use_codec_latent_audio_embedding': True,
        'acoustic_codes_predictor_schedule': [2, 6],
        'parallel_codebook_loss_scale': 0.3,
        'acoustic_codes_predictor_loss_scale': 1.7,
    }
    if compute_auxiliary_codebook_loss is not None:
        overrides['compute_auxiliary_codebook_loss'] = compute_auxiliary_codebook_loss
    with _codec_restore_uses_random_initialized_audio_codec():
        model = NemotronTTSModel(tiny_easy_magpie_cfg(overrides))
    model.eval()
    return model
