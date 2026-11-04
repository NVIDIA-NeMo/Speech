# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from nemo.collections.tts.models.nemotron_tts.v2.model import NemotronTTSModel
from nemo.collections.tts.modules.aligner import init_training_alignment
from tests.collections.tts.modules.nemotron_tts.v2.test_streaming_timestamps_model import _model

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('multiturn', [False, True])
def test_actual_process_batch_combines_losses_and_trains_the_main_backbone_and_teacher(multiturn):
    model = _training_model(multiturn=multiturn)
    with patch.object(model, 'forward', wraps=model.forward) as forward:
        output = model.process_batch(**_inputs(model), mode='val')
    assert output.aligner_loss.requires_grad and output.attention_alignment_loss.requires_grad
    assert torch.isfinite(output.loss)
    expected = (
        model.parallel_codebook_loss_scale * output.codebook_loss
        + model.acoustic_codes_predictor_loss_scale * output.acoustic_codes_predictor_loss
        + model.aligner_loss_weight * output.aligner_loss
        + model.attention_alignment_config['loss_weight'] * output.attention_alignment_loss
    )
    torch.testing.assert_close(output.loss, expected)
    request = forward.call_args.kwargs['attention_alignment']
    assert request.layers == 'middle' and request.heads == (0,)
    assert request.turns[0].key_positions.tolist() == [3, 4]
    assert request.turns[0].query_positions.tolist() == [7, 8, 9, 10]
    assert output.audio_codes_target.shape[1] == model.num_audio_codebooks * model.frame_stacking_factor
    output.loss.backward()
    middle = model.decoder.layers[2].mixer
    assert middle.q_proj.weight.grad is not None and middle.q_proj.weight.grad.abs().sum() > 0
    assert middle.k_proj.weight.grad is not None and middle.k_proj.weight.grad.abs().sum() > 0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.text_aligner.text_encoder.encoder.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.text_aligner.flow.parameters()
    )
    assert not hasattr(model, 'aligned_text_head')


def test_process_batch_acoustic_teacher_does_not_backpropagate_into_tts_or_input_embeddings():
    model = _training_model()
    output = model.process_batch(**_inputs(model), mode='val')
    output.aligner_loss.backward()
    assert model.decoder.get_input_embeddings().weight.grad is None
    assert model.audio_test_embedding.weight.grad is None
    assert model.decoder.layers[2].mixer.q_proj.weight.grad is None
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.text_aligner.text_encoder.parameters()
    )


def test_default_disabled_process_batch_never_invokes_the_teacher():
    model = _training_model(enabled=False)
    with (
        patch(
            'nemo.collections.tts.models.nemotron_tts.v2.model.prepare_attention_alignment_training',
            side_effect=AssertionError('Disabled alignment must preserve the ordinary training path'),
        ),
        patch.object(model, 'forward', wraps=model.forward) as forward,
    ):
        output = model.process_batch(**_inputs(model), mode='val')
    assert 'attention_alignment' not in forward.call_args.kwargs
    assert output.aligner_loss is None and output.attention_alignment_loss is None
    assert not hasattr(model, 'text_aligner')


def test_zero_prior_and_zero_auxiliary_weights_preserve_real_backbone_predictions():
    baseline = _training_model(enabled=False)
    observed = _training_model(enabled=True, prior_strength=0.0, alignment_weight=0.0, aligner_weight=0.0)
    baseline_output = baseline.process_batch(**_inputs(baseline), mode='val')
    observed_output = observed.process_batch(**_inputs(observed), mode='val')
    torch.testing.assert_close(baseline_output.logits, observed_output.logits, rtol=0, atol=0)
    torch.testing.assert_close(baseline_output.loss, observed_output.loss, rtol=0, atol=0)


class _OriginalQuantizer:
    codebook_dim = 4

    def decode(self, indices, input_len):
        values = indices.transpose(0, 1).float().repeat_interleave(2, dim=1) / 16
        mask = torch.arange(values.size(2))[None, :] < input_len[:, None]
        return values * mask[:, None]


def _training_model(enabled=True, *, multiturn=True, prior_strength=0.0, alignment_weight=0.2, aligner_weight=0.3):
    model = _model()
    model.__class__ = NemotronTTSModel
    model._trainer = None
    model._cfg.use_multiturn_dataset = multiturn
    model._cfg.use_attention_alignment = enabled
    model._cfg.aligner_loss_weight = aligner_weight
    model._cfg.aligned_text_aligner = OmegaConf.create(
        dict(
            hidden_channels=8,
            num_flows=1,
            num_text_layers=1,
            coupling_num_layers=1,
            coupling_kernel_size=3,
            text_dropout=0.0,
            dequantization_noise=0.0,
            mas_backend='cpu',
        )
    )
    model._cfg.attention_alignment = OmegaConf.create(
        dict(
            start_step=0,
            ramp_steps=0,
            prior_strength=prior_strength,
            loss_weight=alignment_weight,
        )
    )
    model.context_audio_bos_id, model.context_audio_eos_id = 18, 19
    model.interruption_token_id = 63
    model.text_vocab_size = 64
    model.num_all_tokens_per_codebook = 32
    model.dropout_text_input_prob = 0.0
    model.cfg_unconditional_prob = 0.0
    model.feature_masking = None
    model.parallel_codebook_loss_scale = 0.7
    model.acoustic_codes_predictor_loss_scale = 0.8
    model.label_smoothing = 0.0
    model.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
    model.audio_out_projection = nn.Identity()
    model.final_proj = nn.Linear(16, model.num_all_tokens_per_codebook * 4)
    model._codec_converter = None
    model._codec_model = SimpleNamespace(vector_quantizer=_OriginalQuantizer(), codebook_size=16)
    model.training_modes = list(model.mode_name_to_mode.values())
    init_training_alignment(model, model.cfg)
    return model.train()


def _inputs(model):
    return dict(
        text=torch.tensor([[model.bos_id, 7, 8, model.eos_id]]),
        text_lens=torch.tensor([4]),
        context_text_tokens=torch.tensor([[3, 4]]),
        context_text_tokens_lens=torch.tensor([2]),
        audio_codes=torch.tensor([[[2, 3, 4, 5, 6, 7, 8, 9], [4, 5, 6, 7, 8, 9, 10, 11]]]),
        audio_codes_lens=torch.tensor([8]),
        context_audio_codes=torch.zeros(1, 2, 4, dtype=torch.long),
        context_audio_codes_lens=torch.tensor([4]),
    )
