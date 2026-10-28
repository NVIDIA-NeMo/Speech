# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import PropertyMock, patch

import pytest
import torch

from tests.collections.tts.models.test_easy_magpietts import (
    BPE_TOKENIZER_NAME,
    _make_easy_magpie_model,
    _toy_batch,
    tiny_easy_magpie_cfg,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope='module')
def aligned_model():
    with torch.device('cuda' if torch.cuda.is_available() else 'cpu'):
        return _make_easy_magpie_model(
            tiny_easy_magpie_cfg(
                {
                    'use_aligned_text_head': True,
                    'start_aligner_text_head_steps': 10000,
                    'aligned_text_head_dim': 8,
                    'aligned_text_head_chunk_size': 2,
                    'aligned_text_aligner': {
                        'flow_type': 'sttts',
                        'text_encoder_type': 'relative_transformer',
                        'text_dropout': 0.0,
                        'hidden_channels': 8,
                        'num_flows': 2,
                        'num_text_layers': 1,
                    },
                }
            )
        )


def _alignment_batch(model):
    device = next(model.parameters()).device
    with torch.device(device):
        batch = _toy_batch(model)
        references = [
            model.tokenizer.encode(t, tokenizer_name=BPE_TOKENIZER_NAME) for t in ('hello world', 'good day')
        ]
        tokens = torch.full((2, max(map(len, references))), model.pad_id, dtype=torch.long)
        for row, ids in enumerate(references):
            tokens[row, : len(ids)] = torch.tensor(ids)
        batch['alignment_text_tokens'] = tokens
        batch['alignment_text_lens'] = torch.tensor(list(map(len, references)))
        batch['mode'] = 'val'
    return batch


def test_aligner_and_text_head_train_together(aligned_model):
    model = aligned_model
    model.zero_grad(set_to_none=True)
    batch = _alignment_batch(model)
    with patch.object(type(model), 'global_step', new_callable=PropertyMock, return_value=10000):
        output = model.process_batch(**batch)
    assert torch.isfinite(output.loss)
    assert output.alignment_metrics['valid_turns'] == 2
    assert output.aligned_text_loss > 0
    assert output.aligner_loss is not None
    torch.testing.assert_close(output.loss, output.codebook_loss + output.aligned_text_loss + output.aligner_loss)
    for row, length in enumerate(batch['audio_codes_lens'].tolist()):
        content = set(batch['alignment_text_tokens'][row, : batch['alignment_text_lens'][row]].tolist())
        assert set(output.aligned_text_targets[row, :length].tolist()) <= content
        assert output.aligned_text_targets[row, length] == model.eos_id
    output.loss.backward()
    assert all(p.grad is not None for p in model.text_aligner.parameters())
    assert all(p.grad is not None for p in model.aligned_text_head.parameters())
    assert any(p.grad.abs().sum() > 0 for p in model.text_aligner.parameters())
    assert any(p.grad.abs().sum() > 0 for p in model.aligned_text_head.parameters())
    assert all(p.grad is None for p in model._codec_model.parameters())


def test_curriculum_switches_at_exact_global_step(aligned_model):
    model = aligned_model
    reference_length = int(_alignment_batch(model)['alignment_text_lens'][0])
    with torch.no_grad():
        with patch.object(type(model), 'global_step', new_callable=PropertyMock, return_value=9999):
            warmup = model.process_batch(**_alignment_batch(model))
        with patch.object(type(model), 'global_step', new_callable=PropertyMock, return_value=10000):
            aligned = model.process_batch(**_alignment_batch(model))
    assert warmup.aligned_text_targets[0, reference_length] == model.eos_id
    assert aligned.aligned_text_targets[0, :4].ne(model.eos_id).all()
    assert aligned.aligned_text_targets[0, :4].ne(model.pad_id).all()
    assert warmup.aligner_loss is not None and aligned.aligner_loss is not None


def test_feedback_is_shifted_and_resets_at_turn_boundaries(aligned_model):
    model = aligned_model
    targets = torch.tensor([[11, 11, 12, model.pad_id, 13, 14, model.eos_id]], device=model.device)
    mask = targets.ne(model.pad_id)
    feedback = model._aligned_text_feedback_inputs(targets, mask)
    assert feedback.tolist() == [[model.bos_id, 11, 11, model.pad_id, model.bos_id, 13, 14]]


def test_feedback_resets_at_adjacent_turns(aligned_model):
    model = aligned_model
    targets = torch.tensor([[11, 11, 12, 12]], device=model.device)
    turns = torch.tensor([[0, 0, 2], [0, 2, 4]], device=model.device)
    feedback = model._aligned_text_feedback_inputs(targets, torch.ones_like(targets, dtype=torch.bool), turns)
    assert feedback.tolist() == [[model.bos_id, 11, model.bos_id, 12]]


def test_invalid_overlapping_turn_cannot_reset_valid_feedback(aligned_model):
    model = aligned_model
    targets = torch.tensor([[11, 11, 12, 12]], device=model.device)
    turns = torch.tensor([[0, 0, 4], [0, 2, 2]], device=model.device)
    feedback = model._aligned_text_feedback_inputs(targets, torch.ones_like(targets, dtype=torch.bool), turns)
    assert feedback.tolist() == [[model.bos_id, 11, 11, 12]]


def test_missing_metadata_never_falls_back_to_conditioning_text(aligned_model):
    batch = _alignment_batch(aligned_model)
    del batch['alignment_text_tokens']
    with pytest.raises(ValueError, match='normalized_text'):
        aligned_model.process_batch(**batch)


def test_missing_normalization_excludes_auxiliary_losses(aligned_model):
    batch = _alignment_batch(aligned_model)
    batch['alignment_text_tokens'] = batch['alignment_text_tokens'][:, :0]
    batch['alignment_text_lens'].zero_()
    output = aligned_model.process_batch(**batch)
    assert output.alignment_metrics['valid_turns'] == 0
    assert output.aligner_loss == 0
    assert output.aligned_text_loss == 0
    assert output.aligned_text_targets.eq(aligned_model.pad_id).all()


def test_turn_cannot_align_into_another_items_batch_padding(aligned_model):
    batch = _alignment_batch(aligned_model)
    batch['alignment_turn_indices'] = batch['alignment_text_tokens'].new_tensor([[0, 0, 4], [1, 0, 4]])
    output = aligned_model.process_batch(**batch)
    assert output.alignment_metrics['valid_turns'] == 1
    assert output.aligned_text_targets[1].eq(aligned_model.pad_id).all()


def test_auxiliary_parameters_participate_in_text_only_ddp(aligned_model):
    aligned_model.zero_grad(set_to_none=True)
    aligned_model._aligned_text_zero_loss(torch.zeros((), device=aligned_model.device)).backward()
    for module in (aligned_model.text_aligner, aligned_model.aligned_text_head):
        assert all(p.grad is not None and p.grad.eq(0).all() for p in module.parameters())


def test_aligned_weights_are_saved_and_reloaded(aligned_model):
    state = aligned_model.state_dict()
    assert any(key.startswith('text_aligner.') for key in state)
    assert any(key.startswith('aligned_text_head.') for key in state)
    aligned_model.load_state_dict(state)


def test_pretrained_weights_without_alignment_modules_load(aligned_model, tmp_path):
    state = {
        key: value
        for key, value in aligned_model.state_dict().items()
        if not key.startswith(('text_aligner.', 'aligned_text_head.'))
    }
    checkpoint = tmp_path / 'pre_alignment.ckpt'
    torch.save(state, checkpoint)
    expected = aligned_model.aligned_text_head.output_projection.weight.detach().clone()
    aligned_model.restore_from_pretrained_checkpoint(str(checkpoint))
    torch.testing.assert_close(aligned_model.aligned_text_head.output_projection.weight, expected)


@pytest.mark.parametrize('restoring', [False, True])
def test_unversioned_aligner_config_preserves_compact_architecture(restoring):
    from nemo.collections.tts.models.easy_magpietts_inference import EasyMagpieTTSInferenceModel

    old = {'hidden_channels': 64, 'num_flows': 4, 'num_text_layers': 2}
    resolved = EasyMagpieTTSInferenceModel._resolve_aligned_text_aligner_config(old, restoring=restoring)
    assert resolved['flow_type'] == 'legacy'
    assert resolved['hidden_channels'] == 64 and resolved['num_flows'] == 4
    assert resolved['max_log_scale'] == 2.0
    assert resolved['text_encoder_type'] == 'cnn'


def test_new_aligner_config_and_old_implicit_restoration_are_unambiguous():
    from nemo.collections.tts.models.easy_magpietts_inference import EasyMagpieTTSInferenceModel

    resolve = EasyMagpieTTSInferenceModel._resolve_aligned_text_aligner_config
    new = resolve(None)
    assert (new['flow_type'], new['hidden_channels'], new['num_flows']) == ('sttts', 128, 6)
    assert (new['coupling_num_layers'], new['coupling_kernel_size']) == (4, 5)
    assert new['max_log_scale'] is None
    assert new['text_encoder_type'] == 'relative_transformer'
    assert (new['text_num_heads'], new['text_filter_channels'], new['text_rel_attn_window_size']) == (2, None, 4)
    assert resolve(None, restoring=True)['flow_type'] == 'legacy'
    assert resolve(None, restoring=True)['text_encoder_type'] == 'cnn'
    explicit = {'flow_type': 'sttts', 'text_encoder_type': 'relative_transformer'}
    assert resolve(explicit, restoring=True) == new


@pytest.mark.parametrize('restoring', [False, True])
def test_saved_sttts_flow_without_text_version_keeps_cnn_prior(restoring):
    from nemo.collections.tts.models.easy_magpietts_inference import EasyMagpieTTSInferenceModel

    old = {'flow_type': 'sttts', 'hidden_channels': 128, 'num_flows': 6, 'num_text_layers': 2}
    resolved = EasyMagpieTTSInferenceModel._resolve_aligned_text_aligner_config(old, restoring=restoring)
    assert resolved['flow_type'] == 'sttts' and resolved['text_encoder_type'] == 'cnn'
    assert (resolved['hidden_channels'], resolved['num_flows']) == (128, 6)


def test_explicit_relative_text_config_uses_current_flow_defaults():
    from nemo.collections.tts.models.easy_magpietts_inference import EasyMagpieTTSInferenceModel

    resolved = EasyMagpieTTSInferenceModel._resolve_aligned_text_aligner_config(
        {'text_encoder_type': 'relative_transformer', 'text_num_heads': 4, 'text_dropout': 0.0}
    )
    assert resolved['flow_type'] == 'sttts' and resolved['text_encoder_type'] == 'relative_transformer'
    assert resolved['text_num_heads'] == 4 and resolved['text_dropout'] == 0.0
