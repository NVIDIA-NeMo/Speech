# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import pytest
import torch

from nemo.collections.tts.models.nemotron_tts.v2.model import NemotronTTSModel
from nemo.collections.tts.modules.aligner import (
    collect_aligned_text_head_diagnostics,
    prepare_attention_alignment_training,
)
from nemo.collections.tts.modules.attention_alignment import plot_aligned_text_head_alignment
from tests.collections.tts.modules.nemotron_tts.v2.test_training_model import _inputs, _training_model

pytestmark = pytest.mark.unit


def test_opt_in_diagnostics_leave_actual_losses_unchanged_and_project_in_bounded_chunks():
    model = _training_model(enabled=True, head_enabled=True)
    model.aligned_text_head_chunk_size = 2
    baseline = model.process_batch(**_inputs(model), mode='val')
    with patch.object(model.aligned_text_head, 'forward', wraps=model.aligned_text_head.forward) as project:
        observed = model.process_batch(**_inputs(model), mode='val', return_aligned_text_diagnostics=True)
    assert baseline.aligned_text_diagnostics is None
    assert observed.aligned_text_diagnostics
    torch.testing.assert_close(observed.loss, baseline.loss, rtol=0, atol=0)
    torch.testing.assert_close(observed.logits, baseline.logits, rtol=0, atol=0)
    assert max(call.args[0].shape[-2] for call in project.call_args_list) <= 2
    assert not torch.as_tensor(observed.aligned_text_diagnostics[0]['head_probabilities']).requires_grad


@pytest.mark.parametrize('head_enabled,policy', [(False, False), (True, True)])
def test_disabled_head_and_policy_scoring_never_collect_diagnostics(head_enabled, policy):
    model = _training_model(enabled=False, head_enabled=head_enabled)
    with patch(
        'nemo.collections.tts.models.nemotron_tts.v2.model.collect_aligned_text_head_diagnostics',
        side_effect=AssertionError('Unrequested diagnostic projection'),
    ):
        output = model.process_batch(
            **_inputs(model), mode='val', return_policy_logits=policy, return_aligned_text_diagnostics=True
        )
    assert output.aligned_text_diagnostics is None


def test_probability_columns_keep_repeated_ids_and_off_transcript_predictions():
    model, diagnostics, logits = _diagnostics()
    expected = logits.masked_fill(~model._aligned_text_output_mask, -torch.inf).softmax(-1)
    probabilities = torch.as_tensor(diagnostics['head_probabilities'])
    assert torch.as_tensor(diagnostics['token_ids']).tolist() == [7, 7, 8]
    torch.testing.assert_close(probabilities, expected[:, [7, 7, 8]])
    torch.testing.assert_close(probabilities[:, 0], probabilities[:, 1])
    assert torch.as_tensor(diagnostics['predicted_token_ids']).tolist() == [9] * 4
    torch.testing.assert_close(torch.as_tensor(diagnostics['predicted_probabilities']), expected[:, 9])
    assert diagnostics['frame_seconds'] == pytest.approx(0.08)


def test_alignment_figure_labels_text_and_speech_time_and_preserves_frame_table(tmp_path):
    _, diagnostics, _ = _diagnostics()
    figure, rows = plot_aligned_text_head_alignment(diagnostics, _tokenizer())
    try:
        assert any('bpe' in axis.get_xlabel().lower() for axis in figure.axes)
        assert any('time' in axis.get_ylabel().lower() for axis in figure.axes)
        assert len(rows) == 4 and len(rows[0]) == 9
        assert rows[0][0:3] == pytest.approx([0, 0.0, 0.08])
        assert rows[-1][1:3] == pytest.approx([0.24, 0.32])
        assert [row[3] for row in rows] == [9] * 4
        assert all(row[4] == 'bpe_9' and row[8] for row in rows)
        path = tmp_path / 'aligned_bpe.png'
        figure.savefig(path)
        assert path.stat().st_size > 0
    finally:
        plt.close(figure)


def test_masked_frames_are_gray_without_removing_their_timeline_slots():
    _, diagnostic, _ = _diagnostics(masked_frame=1)
    assert diagnostic['supervised'] == [True, False, True, True]
    assert torch.as_tensor(diagnostic['head_probabilities'])[1].isnan().all()
    assert torch.as_tensor(diagnostic['mas_alignment'])[1].isnan().all()
    assert diagnostic['start_frame'] == 0 and diagnostic['end_frame'] == 4
    assert len(diagnostic['predicted_token_ids']) == 4


def test_supervised_audio_eos_keeps_its_own_frame_and_table_label():
    model, diagnostic, _ = _diagnostics(include_eos=True)
    assert diagnostic['end_frame'] == 5 and diagnostic['target_token_ids'][-1] == model.eos_id
    assert diagnostic['supervised'][-1]
    assert torch.as_tensor(diagnostic['mas_alignment'])[-1].isnan().all()
    figure, rows = plot_aligned_text_head_alignment(diagnostic, _tokenizer())
    try:
        assert rows[-1][1:3] == pytest.approx([0.32, 0.4])
        assert rows[-1][6:9] == [model.eos_id, '<EOS>', True]
    finally:
        plt.close(figure)


def test_long_turn_bounds_the_figure_but_retains_every_frame_prediction_and_table_row():
    model = _training_model(enabled=False, head_enabled=True)
    frames, count = 1031, 257
    tokens = (torch.arange(count) % 59 + 1)[None, :]
    path = (torch.arange(frames) * count // frames)[None, :]
    targets = tokens.gather(1, path)
    compact = (tokens, torch.tensor([count]), torch.tensor([[0, 0, frames]]), torch.arange(count)[None, :])
    alignment = SimpleNamespace(
        teacher=SimpleNamespace(token_indices=path), metrics={'aligned_text_curriculum_active': True}
    )
    with patch('nemo.collections.tts.modules.aligner.compact_conditioning_turns', return_value=compact):
        diagnostic = collect_aligned_text_head_diagnostics(
            model,
            torch.zeros(1, frames, 16),
            alignment,
            tokens,
            torch.tensor([count]),
            torch.tensor([frames * model.frame_stacking_factor]),
            targets,
            torch.ones_like(targets, dtype=torch.bool),
        )[0]
    assert diagnostic['mas_alignment'].shape == diagnostic['head_probabilities'].shape == (1024, 256)
    assert len(diagnostic['token_ids']) == count
    assert len(diagnostic['predicted_token_ids']) == len(diagnostic['target_token_ids']) == frames
    figure, rows = plot_aligned_text_head_alignment(diagnostic, _tokenizer())
    try:
        assert len(rows) == frames
        assert rows[-1][0] == frames - 1
        assert rows[-1][2] == pytest.approx(frames * diagnostic['frame_seconds'])
    finally:
        plt.close(figure)


@pytest.mark.parametrize('head_enabled,wandb_enabled', [(False, True), (True, False)])
def test_logging_skips_plotting_without_enabled_head_and_wandb(head_enabled, wandb_enabled):
    logger = _Logger() if wandb_enabled else object()
    model = SimpleNamespace(use_aligned_text_head=head_enabled, loggers=[logger])
    with (
        patch('nemo.collections.tts.models.nemotron_tts.v2.model.WandbLogger', _Logger),
        patch(
            'nemo.collections.tts.models.nemotron_tts.v2.model.plot_aligned_text_head_alignment',
            side_effect=AssertionError('Disabled diagnostics rendered a figure'),
        ),
    ):
        assert (
            NemotronTTSModel._log_aligned_text_head_examples(model, SimpleNamespace(aligned_text_diagnostics=[{}]))
            == {}
        )


def test_wandb_media_contains_figure_and_repeated_token_table_and_closes_figure():
    _, diagnostics, _ = _diagnostics()
    model = SimpleNamespace(use_aligned_text_head=True, loggers=[_Logger()], tokenizer=_tokenizer(), global_step=17)
    before = plt.get_fignums()
    with (
        patch('nemo.collections.tts.models.nemotron_tts.v2.model.WandbLogger', _Logger),
        patch('nemo.collections.tts.models.nemotron_tts.v2.model.wandb.Image', return_value='image') as image,
        patch('nemo.collections.tts.models.nemotron_tts.v2.model.wandb.Table', return_value='table') as table,
    ):
        media = NemotronTTSModel._log_aligned_text_head_examples(
            model, SimpleNamespace(aligned_text_diagnostics=[diagnostics])
        )
    assert image.call_count == table.call_count == 1
    assert 'image' in media.values() and 'table' in media.values()
    assert plt.get_fignums() == before


@pytest.mark.parametrize(
    'head,rank,batch_idx,wandb_enabled',
    [(True, 0, 0, True), (False, 0, 0, True), (True, 1, 0, True), (True, 0, 1, True), (True, 0, 0, False)],
)
def test_validation_collects_only_first_rank_zero_wandb_batch(head, rank, batch_idx, wandb_enabled):
    output = SimpleNamespace(
        **{key: torch.zeros(()) for key in ['loss', 'codebook_loss', 'acoustic_codes_predictor_loss']}
    )
    for key in [
        'logits',
        'audio_codes_target',
        'audio_codes_lens_target',
        'context_audio_codes',
        'context_audio_codes_lens',
    ]:
        setattr(output, key, None)
    model = SimpleNamespace(
        global_rank=rank,
        local_rank=rank,
        trainer=SimpleNamespace(world_size=2),
        use_aligned_text_head=head,
        loggers=[_Logger() if wandb_enabled else object()],
        process_batch=Mock(return_value=output),
        _log_alignment_losses=Mock(),
        log_val_audio_example=Mock(return_value={}),
        _log_aligned_text_head_examples=Mock(return_value={}),
        phoneme_tokenizer=None,
        run_val_inference=False,
        validation_step_outputs=[],
    )
    batch = {
        key: torch.zeros(1, 1, dtype=torch.long)
        for key in ['text', 'context_text_tokens', 'audio_codes', 'context_audio_codes']
    }
    batch.update(
        {
            key: torch.ones(1, dtype=torch.long)
            for key in ['text_lens', 'context_text_tokens_lens', 'audio_codes_lens', 'context_audio_codes_lens']
        }
    )
    with patch('nemo.collections.tts.models.nemotron_tts.v2.model.WandbLogger', _Logger):
        NemotronTTSModel.validation_step(model, batch, batch_idx)
    assert model.process_batch.call_args.kwargs.get('return_aligned_text_diagnostics', False) is (
        head and rank == 0 and batch_idx == 0 and wandb_enabled
    )


class _Logger:
    def __init__(self):
        self.experiment = SimpleNamespace(log=Mock())


def _tokenizer():
    return SimpleNamespace(
        convert_ids_to_tokens=lambda value: f'bpe_{value}', decode=lambda values: ' '.join(f'bpe_{v}' for v in values)
    )


def _diagnostics(masked_frame=None, include_eos=False):
    model = _training_model(enabled=False, head_enabled=True)
    model.aligned_text_head_chunk_size = 2
    model._cfg.use_multiturn_dataset = not include_eos
    text = torch.tensor([[model.bos_id, 7, 7, 8, model.eos_id]])
    audio = _inputs(model)
    steps = 5 if include_eos else 4
    targets = torch.zeros(1, 4, steps, dtype=torch.long)
    if include_eos:
        targets[:, :, -1] = model.audio_eos_id
    alignment = prepare_attention_alignment_training(
        model,
        audio['audio_codes'],
        audio['audio_codes_lens'],
        targets,
        text,
        torch.tensor([5]),
        torch.tensor([2]),
        torch.tensor([7]),
    )
    hidden = torch.zeros(1, steps, 16)
    hidden[:, :, 0] = 1
    with torch.no_grad():
        for parameter in model.aligned_text_head.parameters():
            parameter.zero_()
        model.aligned_text_head.input_projection.weight[0, 0] = 1
        model.aligned_text_head.output_projection.weight[[7, 8, 9], 0] = torch.tensor([1.0, 2.0, 5.0])
    logits = model.aligned_text_head(hidden[0]).detach()
    mask = alignment.aligned_text_mask.clone()
    if masked_frame is not None:
        mask[:, masked_frame] = False
    result = collect_aligned_text_head_diagnostics(
        model,
        hidden.requires_grad_(),
        alignment,
        text,
        torch.tensor([5]),
        audio['audio_codes_lens'],
        alignment.aligned_text_targets,
        mask,
    )
    assert len(result) == 1
    return model, result[0], logits
