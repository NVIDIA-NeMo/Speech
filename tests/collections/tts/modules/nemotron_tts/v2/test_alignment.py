# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from nemo.collections.tts.modules.aligner import (
    TinyGlowTTSAligner,
    alignment_schedule,
    compact_conditioning_turns,
    compact_speaker_reference,
    decode_alignment_features,
    init_training_alignment,
    prepare_attention_alignment_training,
    resolve_attention_alignment_config,
)

pytestmark = pytest.mark.unit


def test_alignment_disabled_by_default_constructs_no_teacher():
    model = SimpleNamespace()
    init_training_alignment(model, {})
    assert model.use_attention_alignment is False
    assert not hasattr(model, 'text_aligner')


@pytest.mark.parametrize('stack', [1, 2])
def test_training_alignment_defaults_to_eighty_channels_independently_of_codec_and_stacking(stack):
    model = nn.Module()
    model.cfg = OmegaConf.create(
        dict(
            embedding_dim=6,
            use_attention_alignment=True,
            aligned_text_aligner=dict(
                hidden_channels=8,
                num_flows=1,
                num_text_layers=1,
                coupling_num_layers=1,
                speaker_hidden_channels=4,
                speaker_embedding_dim=4,
                speaker_num_layers=1,
            ),
        )
    )
    model._is_model_being_restored = lambda: False
    model._codec_model = SimpleNamespace(vector_quantizer=SimpleNamespace(codebook_dim=96))
    model.frame_stacking_factor = stack
    model.decoder = SimpleNamespace(config=SimpleNamespace(layers_block_type=['attention'], num_attention_heads=2))
    init_training_alignment(model, model.cfg)
    assert model.text_aligner.audio_channels == 80
    assert model.text_aligner.input_audio_channels == 96 * stack
    assert model.text_aligner.speaker_encoder is not None
    assert model.cfg.aligned_text_aligner.audio_channels == 80
    assert model.cfg.aligned_text_aligner.input_audio_channels == 96 * stack


def test_singleturn_retains_adjacent_repeated_bpes_and_actual_key_positions():
    model = _model()
    text = torch.tensor([[model.bos_id, 7, 7, 8, model.eos_id, model.pad_id]])
    compact, lengths, turns, keys = compact_conditioning_turns(
        model, text, torch.tensor([5]), torch.tensor([12]), False
    )
    assert compact.tolist() == [[7, 7, 8]]
    assert lengths.tolist() == [3]
    assert turns.tolist() == [[0, 0, 6]]
    assert keys.tolist() == [[1, 2, 3]]


def test_multiturn_uses_bos_audio_start_and_interruption_end_not_text_eos():
    model = _model()
    text = torch.tensor(
        [
            [
                model.bos_id,
                7,
                8,
                model.eos_id,
                model.pad_id,
                model.pad_id,
                model.interruption_token_id,
                model.bos_id,
                9,
                model.eos_id,
                model.pad_id,
                model.pad_id,
            ]
        ]
    )
    compact, lengths, turns, keys = compact_conditioning_turns(
        model, text, torch.tensor([12]), torch.tensor([24]), True
    )
    assert compact.tolist() == [[7, 8], [9, model.pad_id]]
    assert lengths.tolist() == [2, 1]
    assert turns.tolist() == [[0, 0, 6], [0, 7, 12]]
    assert keys.tolist() == [[1, 2], [8, -1]]


def test_multiturn_without_bos_and_cropped_final_marker_preserves_last_word():
    model = _model()
    text = torch.tensor(
        [
            [
                model.pad_id,
                7,
                8,
                model.eos_id,
                model.pad_id,
                model.interruption_token_id,
                model.pad_id,
                9,
                10,
                model.eos_id,
                model.pad_id,
            ]
        ]
    )
    compact, lengths, turns, keys = compact_conditioning_turns(
        model, text, torch.tensor([11]), torch.tensor([21]), True
    )
    assert compact.tolist() == [[7, 8], [9, 10]]
    assert lengths.tolist() == [2, 2]
    assert turns.tolist() == [[0, 1, 5], [0, 7, 11]]
    assert keys.tolist() == [[1, 2], [7, 8]]


def test_empty_turn_metadata_keeps_a_valid_empty_tensor_contract():
    model = _model()
    compact, lengths, turns, keys = compact_conditioning_turns(
        model, torch.full((2, 4), model.pad_id), torch.tensor([4, 4]), torch.tensor([8, 8]), True
    )
    assert compact.shape == keys.shape == (0, 0)
    assert lengths.shape == (0,) and turns.shape == (0, 3)


def test_empty_acoustic_observations_keep_zero_teacher_loss_with_a_valid_gradient_graph():
    model = _model()
    output = prepare_attention_alignment_training(
        model,
        torch.zeros(1, 2, 0, dtype=torch.long),
        torch.tensor([0]),
        torch.zeros(1, 4, 0, dtype=torch.long),
        torch.tensor([[7, 8, model.eos_id]]),
        torch.tensor([3]),
        torch.tensor([2]),
        torch.tensor([7]),
    )
    assert output.guidance is None
    assert output.teacher.token_indices.shape == (1, 0)
    assert output.teacher.loss.item() == 0.0
    output.teacher.loss.backward()
    assert model.text_aligner.weight.grad.item() == 0.0


def test_training_schedule_warms_flow_then_ramps_loss_and_anneals_only_prior():
    config = resolve_attention_alignment_config(dict(start_step=10, ramp_steps=2, prior_anneal_steps=4))
    assert alignment_schedule(config, 9) == (0.0, 0.0)
    assert alignment_schedule(config, 11) == (0.5, 0.5)
    assert alignment_schedule(config, 12) == (1.0, 1.0)
    assert alignment_schedule(config, 14) == (1.0, 0.5)
    assert alignment_schedule(config, 16) == (1.0, 0.0)


@pytest.mark.parametrize(
    'config',
    [
        dict(ramp_steps=-1),
        dict(band_width=0),
        dict(prior_strength=1.1),
        dict(layers='extra'),
        dict(heads=[0, 0]),
        dict(unknown=True),
    ],
)
def test_invalid_training_alignment_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        resolve_attention_alignment_config(config)


def test_teacher_trains_with_detached_inputs_and_exact_unshifted_query_grid():
    model = _model()
    text = torch.tensor([[model.bos_id, 7, 8, model.eos_id]])
    codes = torch.zeros(1, 2, 8, dtype=torch.long)
    output = prepare_attention_alignment_training(
        model,
        codes,
        torch.tensor([8]),
        torch.zeros(1, 4, 5, dtype=torch.long),
        text,
        torch.tensor([4]),
        torch.tensor([3]),
        torch.tensor([8]),
    )
    assert len(output.guidance.turns) == 1
    turn = output.guidance.turns[0]
    assert turn.query_positions.tolist() == [8, 9, 10, 11]
    assert turn.key_positions.tolist() == [4, 5]
    assert turn.token_indices.tolist() == [0, 0, 1, 1]
    assert output.teacher.token_indices.tolist() == [[0, 0, 1, 1, -1]]
    output.teacher.loss.backward()
    assert model.text_aligner.weight.grad is not None
    assert model.embedding.weight.grad is None
    assert not model.text_aligner.last_audio.requires_grad
    assert not model.text_aligner.last_text.requires_grad


def test_text_dropout_suppresses_attention_loss_but_keeps_teacher_training():
    model = _model()
    output = prepare_attention_alignment_training(
        model,
        torch.zeros(1, 2, 8, dtype=torch.long),
        torch.tensor([8]),
        torch.zeros(1, 4, 5, dtype=torch.long),
        torch.tensor([[7, 8, model.eos_id]]),
        torch.tensor([3]),
        torch.tensor([3]),
        torch.tensor([8]),
        dropout_conditioning=True,
    )
    assert output.guidance is None
    assert output.teacher.loss.requires_grad
    assert output.metrics['valid_turns'].item() == 1


def test_future_keys_are_excluded_from_attention_supervision():
    model = _model()
    output = prepare_attention_alignment_training(
        model,
        torch.zeros(1, 2, 8, dtype=torch.long),
        torch.tensor([8]),
        torch.zeros(1, 4, 5, dtype=torch.long),
        torch.tensor([[7, 8, model.eos_id]]),
        torch.tensor([3]),
        torch.tensor([10]),
        torch.tensor([0]),
    )
    assert output.guidance is None
    assert output.metrics['causal_frames'].item() == 0
    assert output.teacher.loss.requires_grad


def test_codec_latent_stacking_uses_original_quantizer_and_frozen_observations():
    model = _model()
    features = decode_alignment_features(model, torch.zeros(1, 2, 4, dtype=torch.long), torch.tensor([4]))
    assert features.tolist() == [[[0.0, 1.0, 10.0, 11.0], [2.0, 3.0, 12.0, 13.0]]]
    assert not features.requires_grad


def test_actual_flow_and_relative_text_encoder_receive_gradients_but_input_embeddings_do_not():
    model = _model()
    model.text_aligner = TinyGlowTTSAligner(
        audio_channels=4,
        text_channels=4,
        hidden_channels=8,
        num_flows=1,
        num_text_layers=1,
        coupling_num_layers=1,
        coupling_kernel_size=3,
        text_dropout=0.0,
        dequantization_noise=0.0,
        mas_backend='cpu',
    )
    output = prepare_attention_alignment_training(
        model,
        torch.zeros(1, 2, 8, dtype=torch.long),
        torch.tensor([8]),
        torch.zeros(1, 4, 5, dtype=torch.long),
        torch.tensor([[7, 8, model.eos_id]]),
        torch.tensor([3]),
        torch.tensor([3]),
        torch.tensor([8]),
    )
    assert output.teacher.valid_turn_count.item() == 1
    assert torch.isfinite(output.teacher.loss)
    output.teacher.loss.backward()
    assert model.embedding.weight.grad is None
    encoder = model.text_aligner.text_encoder.encoder
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in encoder.parameters())
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.text_aligner.flow.parameters()
    )


def test_dummy_speaker_reference_is_not_interpreted_as_original_codec_audio():
    model = _model()
    # Dummy converted-vocabulary IDs can overlap valid original-codec numbers.
    model.context_audio_bos_id, model.context_audio_eos_id = 14, 15
    codes = torch.tensor([[[14, 15], [14, 15]]])
    packed, lengths = compact_speaker_reference(model, codes, torch.tensor([2]))
    assert packed.shape == (1, 2, 0) and lengths.tolist() == [0]
    model.text_aligner = TinyGlowTTSAligner(
        audio_channels=4,
        text_channels=4,
        hidden_channels=8,
        num_flows=1,
        num_text_layers=1,
        coupling_num_layers=1,
        speaker_hidden_channels=4,
        speaker_embedding_dim=4,
        speaker_num_layers=1,
        use_speaker_conditioning=True,
        mas_backend='cpu',
    )
    output = prepare_attention_alignment_training(
        model,
        torch.zeros(1, 2, 8, dtype=torch.long),
        torch.tensor([8]),
        torch.zeros(1, 4, 5, dtype=torch.long),
        torch.tensor([[7, 8, model.eos_id]]),
        torch.tensor([3]),
        torch.tensor([3]),
        torch.tensor([8]),
        reference_codes=codes,
        reference_lens=torch.tensor([2]),
    )
    assert torch.isfinite(output.teacher.loss)
    assert output.teacher.valid_turn_count.item() == 1


def test_control_frames_do_not_become_acoustic_teacher_observations():
    model = _model()
    codes = torch.zeros(1, 2, 8, dtype=torch.long)
    codes[:, :, 3] = 130
    output = prepare_attention_alignment_training(
        model,
        codes,
        torch.tensor([8]),
        torch.zeros(1, 4, 5, dtype=torch.long),
        torch.tensor([[7, 8, model.eos_id]]),
        torch.tensor([3]),
        torch.tensor([3]),
        torch.tensor([8]),
    )
    assert output.teacher.valid_turn_count.item() == 0
    assert output.guidance is None


class _Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.audio_channels = 4
        self.input_audio_channels = 4
        self.use_speaker_conditioning = False

    def forward(self, audio, text, audio_lens, text_lens):
        self.last_audio, self.last_text = audio, text
        path = torch.div(
            torch.arange(audio.size(1))[None, :] * text_lens[:, None], audio_lens[:, None], rounding_mode='floor'
        ).clamp_max(text.size(1) - 1)
        durations = torch.zeros(text.size(0), text.size(1), dtype=torch.long)
        durations.scatter_add_(1, path, torch.ones_like(path))
        return SimpleNamespace(
            loss=self.weight.square(),
            token_indices=path,
            durations=durations,
            valid_turns=torch.ones(text.size(0), dtype=torch.bool),
        )


class _Quantizer:
    def decode(self, indices, input_len):
        batch, frames = indices.size(1), indices.size(2)
        return (
            torch.arange(frames, dtype=torch.float)[None, None].expand(batch, 2, -1)
            + torch.tensor([0.0, 10.0])[None, :, None]
        )


def _model():
    embedding = nn.Embedding(32, 4)
    return SimpleNamespace(
        cfg={},
        pad_id=0,
        bos_id=29,
        eos_id=30,
        interruption_token_id=31,
        text_vocab_size=32,
        frame_stacking_factor=2,
        global_step=0,
        embedding=embedding,
        embed_text_tokens=lambda ids, text_lens: embedding(ids),
        text_aligner=_Teacher(),
        _codec_model=SimpleNamespace(vector_quantizer=_Quantizer(), codebook_size=128),
        context_audio_bos_id=130,
        context_audio_eos_id=131,
        attention_alignment_config=resolve_attention_alignment_config(dict(start_step=0, ramp_steps=0)),
    )
