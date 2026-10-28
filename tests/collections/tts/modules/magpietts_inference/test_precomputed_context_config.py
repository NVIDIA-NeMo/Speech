# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import Mock, patch

import pytest

from nemo.collections.tts.modules.magpietts_inference.evaluate_generated_audio import load_evalset_config
from nemo.collections.tts.modules.magpietts_inference.inference import (
    EasyMagpieInferenceConfig,
    EasyMagpieInferenceRunner,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('key', ['speaker_a', '__zero__'])
def test_evalset_accepts_precomputed_context_key(tmp_path, key):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('', encoding='utf-8')
    info = {'manifest_path': 'manifest.json', 'audio_dir': '.', 'precomputed_context_audio_embedding_key': key}
    config = tmp_path / 'evalset.json'
    config.write_text(json.dumps({'test': info}), encoding='utf-8')

    loaded = load_evalset_config(str(config), dataset_base_path=tmp_path)

    assert loaded['test']['precomputed_context_audio_embedding_key'] == key
    assert loaded['test']['manifest_path'] == str(manifest)


@pytest.mark.parametrize('key', [None, '', ' ', ' speaker_a', 123])
def test_evalset_rejects_invalid_precomputed_context_key(tmp_path, key):
    config = tmp_path / 'evalset.json'
    config.write_text(
        json.dumps(
            {
                'test': {
                    'manifest_path': 'manifest.json',
                    'audio_dir': '.',
                    'precomputed_context_audio_embedding_key': key,
                }
            }
        ),
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='precomputed_context_audio_embedding_key'):
        load_evalset_config(str(config), dataset_base_path=tmp_path)


def test_embedding_table_load_uses_weights_only(tmp_path):
    table_path = tmp_path / 'embeddings.pt'
    config = EasyMagpieInferenceConfig(precomputed_context_audio_embeddings_path=str(table_path))

    with (
        patch.object(EasyMagpieInferenceRunner, '_configure_tokenizer'),
        patch(
            'nemo.collections.tts.modules.magpietts_inference.inference.torch.load',
            return_value={'audio_embeddings': {}},
        ) as mock_load,
    ):
        runner = EasyMagpieInferenceRunner(Mock(), config)

    mock_load.assert_called_once_with(str(table_path), map_location='cpu', weights_only=True)
    assert runner._precomputed_audio_embeddings == {}
