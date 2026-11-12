# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from nemo.collections.tts.data.text_to_speech_dataset import TextToSpeechDataset

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('ignore_manifest_language', [False, True])
@pytest.mark.parametrize('dataset_language', ['en', None])
@pytest.mark.parametrize('manifest_language', ['fr', None, 'missing'])
def test_manifest_language_precedence(tmp_path, ignore_manifest_language, dataset_language, manifest_language):
    entry = {'audio_filepath': 'unused.wav', 'duration': 1.0, 'text': 'hello'}
    if manifest_language != 'missing':
        entry['language'] = manifest_language
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(entry) + '\n', encoding='utf-8')

    dataset = TextToSpeechDataset(
        dataset_meta={
            'test': {
                'manifest_path': manifest,
                'audio_dir': tmp_path,
                'language': dataset_language,
                'ignore_manifest_language': ignore_manifest_language,
            }
        },
        sample_rate=24000,
        text_tokenizer=lambda text: [],
    )

    expected = dataset_language
    if not ignore_manifest_language and manifest_language != 'missing':
        expected = manifest_language
    assert dataset.data_samples[0].language == expected


def test_language_override_is_per_dataset_and_disabled_by_default(tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(
        json.dumps({'audio_filepath': 'unused.wav', 'duration': 1.0, 'text': 'hello', 'language': 'fr'}) + '\n',
        encoding='utf-8',
    )
    meta = {'manifest_path': manifest, 'audio_dir': tmp_path, 'language': 'en'}
    dataset = TextToSpeechDataset(
        dataset_meta={'default': meta, 'override': {**meta, 'ignore_manifest_language': True}},
        sample_rate=24000,
        text_tokenizer=lambda text: [],
    )
    assert [sample.language for sample in dataset.data_samples] == ['fr', 'en']
