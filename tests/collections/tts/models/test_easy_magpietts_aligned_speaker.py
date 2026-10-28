# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch

from tests.collections.tts.models.test_easy_magpietts import _make_easy_magpie_model, tiny_easy_magpie_cfg
from tests.collections.tts.models.test_easy_magpietts_aligned_text import _alignment_batch

pytestmark = pytest.mark.unit


@pytest.fixture(scope='module')
def speaker_model():
    with torch.device('cuda' if torch.cuda.is_available() else 'cpu'):
        model = _make_easy_magpie_model(
            tiny_easy_magpie_cfg(
                {
                    'use_aligned_text_head': True,
                    'start_aligner_text_head_steps': 0,
                    'aligned_text_head_dim': 8,
                    'aligned_text_aligner': {
                        'flow_type': 'sttts',
                        'hidden_channels': 8,
                        'num_flows': 2,
                        'coupling_num_layers': 2,
                        'text_encoder_type': 'relative_transformer',
                        'text_dropout': 0.0,
                        'use_speaker_conditioning': True,
                        'speaker_hidden_channels': 8,
                        'speaker_embedding_dim': 8,
                        'speaker_num_layers': 1,
                    },
                }
            )
        )
        with torch.no_grad():
            for coupling in model.text_aligner.flow.couplings:
                coupling.output_projection.weight.normal_(std=0.02)
        return model


def test_reference_uses_raw_codec_ids_and_encoder_learns_through_model_loss(speaker_model):
    model = speaker_model
    model.zero_grad(set_to_none=True)
    batch = _alignment_batch(model)
    expected, lengths = model._decode_aligner_speaker_reference(
        batch['context_audio_codes'], batch['context_audio_codes_lens']
    )
    encoder = model.text_aligner.speaker_encoder
    with patch.object(encoder, 'forward', wraps=encoder.forward) as encode:
        output = model.process_batch(**batch)
    assert encode.call_count == 1
    features, actual_lengths = encode.call_args.args
    torch.testing.assert_close(features, expected)
    torch.testing.assert_close(actual_lengths, lengths)
    assert not features.requires_grad
    output.loss.backward()
    assert torch.isfinite(output.loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())
    assert encoder.output_projection.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model._codec_model.parameters())


def test_replaced_or_empty_references_are_zero_conditioning(speaker_model):
    batch = _alignment_batch(speaker_model)
    features, lengths = speaker_model._decode_aligner_speaker_reference(
        batch['context_audio_codes'],
        batch['context_audio_codes_lens'],
        torch.tensor([False, True], device=speaker_model.device),
    )
    assert lengths[0] == 0
    embeddings = speaker_model.text_aligner.encode_speaker_reference(features, lengths)
    assert embeddings[0].eq(0).all() and torch.isfinite(embeddings).all()
    batch['context_audio_codes'] = batch['context_audio_codes'][:, :, :0]
    batch['context_audio_codes_lens'].zero_()
    output = speaker_model.process_batch(**batch)
    assert torch.isfinite(output.loss) and output.alignment_metrics['valid_turns'] == 2


def test_missing_reference_does_not_fall_back_to_target_audio(speaker_model):
    assert speaker_model._decode_aligner_speaker_reference(None, None) == (None, None)


def test_saved_aligner_defaults_keep_speaker_conditioning_disabled():
    from nemo.collections.tts.models.easy_magpietts_inference import EasyMagpieTTSInferenceModel

    resolve = EasyMagpieTTSInferenceModel._resolve_aligned_text_aligner_config
    for previous in [None, {'flow_type': 'sttts'}, {'hidden_channels': 64}]:
        assert resolve(previous)['use_speaker_conditioning'] is False
    assert resolve({'flow_type': 'sttts', 'use_speaker_conditioning': True})['use_speaker_conditioning'] is True
