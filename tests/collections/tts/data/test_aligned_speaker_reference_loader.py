# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from nemo.collections.tts.data.text_to_speech_dataset_lhotse_multiturn import MagpieTTSLhotseMultiturnDataset

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('cached_codes', [True, False])
def test_shuffled_references_are_marked_without_disabling_unmodified_rows(cached_codes, monkeypatch):
    dataset = _dataset()
    features, context_key, lengths_key = _features(cached_codes)
    originals = list(features[context_key])
    monkeypatch.setattr('nemo.collections.tts.data.text_to_speech_dataset_lhotse_multiturn.random.random', lambda: 0.0)

    dataset._maybe_shuffle_context_audio(features)

    assert features[context_key][0] is originals[1]
    assert features[context_key][1] is originals[0]
    assert features[context_key][2] is originals[2]
    assert features[context_key][3] is originals[3]
    assert features[lengths_key] == [5, 4, 6, 1]
    assert features['alignment_speaker_reference_valid'] == [False, False, True, True]

    batch = {}
    dataset._add_optional_batch_fields(batch, {}, features)
    assert batch['alignment_speaker_reference_valid'].dtype == torch.bool
    assert batch['alignment_speaker_reference_valid'].shape == (4,)
    assert batch['alignment_speaker_reference_valid'].tolist() == [False, False, True, True]


@pytest.mark.parametrize('cached_codes', [True, False])
def test_feature_disabled_keeps_existing_reference_shuffle_and_batch_fields(cached_codes, monkeypatch):
    dataset = _dataset(aligned=False)
    features, context_key, _ = _features(cached_codes)
    originals = list(features[context_key])
    monkeypatch.setattr('nemo.collections.tts.data.text_to_speech_dataset_lhotse_multiturn.random.random', lambda: 0.0)

    dataset._maybe_shuffle_context_audio(features)

    assert features[context_key][0] is originals[1]
    assert features[context_key][1] is originals[0]
    assert 'alignment_speaker_reference_valid' not in features
    batch = {}
    dataset._add_optional_batch_fields(batch, {}, features)
    assert 'alignment_speaker_reference_valid' not in batch


@pytest.mark.parametrize('reason', ['disabled', 'validation', 'one_eligible'])
def test_no_reference_mask_is_added_when_contexts_are_not_replaced(reason, monkeypatch):
    dataset = _dataset()
    features, context_key, _ = _features(True)
    originals = list(features[context_key])
    if reason == 'disabled':
        dataset.context_audio_shuffle_batch_prob = 0.0
    elif reason == 'validation':
        dataset.dataset_type = 'val'
    else:
        features['has_text_context'][1] = True
    monkeypatch.setattr('nemo.collections.tts.data.text_to_speech_dataset_lhotse_multiturn.random.random', lambda: 0.0)

    dataset._maybe_shuffle_context_audio(features)

    assert all(actual is expected for actual, expected in zip(features[context_key], originals))
    assert 'alignment_speaker_reference_valid' not in features
    batch = {}
    dataset._add_optional_batch_fields(batch, {}, features)
    assert 'alignment_speaker_reference_valid' not in batch


def _dataset(aligned=True):
    dataset = MagpieTTSLhotseMultiturnDataset.__new__(MagpieTTSLhotseMultiturnDataset)
    dataset.dataset_type = 'train'
    dataset.context_audio_shuffle_batch_prob = 1.0
    dataset.codec_model_samples_per_frame = 1
    dataset.use_aligned_text_head = aligned
    dataset.phoneme_tokenizer = None
    dataset.use_text_conditioning_tokenizer = False
    dataset.include_align_prior = False
    return dataset


def _features(cached_codes):
    lengths = [4, 5, 6, 1]
    context_key = 'context_audio_codes' if cached_codes else 'context_audio'
    lengths_key = 'context_audio_codes_lens' if cached_codes else 'context_audio_lens'
    features = {
        'languages': ['en', 'fr', 'en', 'fr'],
        'has_text_context': [False, False, True, False],
        'target_codes': [],
        'source_codes': [],
        'audio_16khz': [],
        'align_priors': [],
        'rewards': [],
        'context_audio': [],
        'context_audio_lens': [],
        'context_audio_codes': [],
        'context_audio_codes_lens': [],
    }
    features[context_key] = [
        torch.full(
            (length, 2) if cached_codes else (length,),
            index + 1,
            dtype=torch.long if cached_codes else torch.float32,
        )
        for index, length in enumerate(lengths)
    ]
    features[lengths_key] = lengths
    return features, context_key, lengths_key
