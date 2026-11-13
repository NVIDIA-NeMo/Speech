# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf
from transformers import ByT5Tokenizer

from nemo.collections.tts.data.text_to_speech_dataset_lhotse import setup_tokenizers


@pytest.fixture
def custom_repository(tmp_path):
    (tmp_path / 'tokenizer_config.json').write_text(
        json.dumps(
            {
                'tokenizer_class': 'CustomTokenizer',
                'auto_map': {'AutoTokenizer': ['tokenization_custom.CustomTokenizer', None]},
            }
        )
    )
    (tmp_path / 'config.json').write_text(
        json.dumps(
            {
                'model_type': 'custom_backend',
                'auto_map': {'AutoConfig': 'configuration_custom.CustomConfig'},
            }
        )
    )
    for filename in ['tokenization_custom.py', 'configuration_custom.py']:
        (tmp_path / filename).write_text("raise RuntimeError('unexpected module import')\n")
    return str(tmp_path)


@pytest.mark.unit
@pytest.mark.parametrize('mode', ['train', 'test'])
def test_tts_tokenizers_reject_custom_repository_code(custom_tokenizer_repository, mode):
    custom_repository = custom_tokenizer_repository
    config = OmegaConf.create(
        {'text': {'_target_': 'AutoTokenizer', 'pretrained_model': custom_repository, 'trust_remote_code': True}}
    )
    with pytest.raises(ValueError, match='trust_remote_code'):
        setup_tokenizers(config, mode=mode)


@pytest.mark.unit
def test_tts_builtin_tokenizer_loads_locally(tmp_path):
    ByT5Tokenizer().save_pretrained(tmp_path)
    config = OmegaConf.create({'text': {'_target_': 'AutoTokenizer', 'pretrained_model': str(tmp_path)}})
    tokenizer = setup_tokenizers(config)
    assert isinstance(tokenizer.tokenizers['text'], ByT5Tokenizer)


@pytest.mark.unit
def test_easy_magpie_backend_rejects_custom_repository_code(custom_repository):
    from nemo.collections.tts.models.easy_magpietts_inference import EasyMagpieTTSInferenceModel

    cfg = OmegaConf.create(
        {
            'codecmodel_path': 'unused.nemo',
            'text_tokenizers': {'text': {}},
            'hidden_dim': 8,
            'embedding_dim': 8,
            'transformer_hf_backend': custom_repository,
            'trust_remote_code': True,
        }
    )
    codec = _mock_codec()
    tokenizer = MagicMock()
    tokenizer.tokens = ['a', 'b']
    tokenizer.pad = 0
    with (
        patch('nemo.collections.tts.models.easy_magpietts_inference.AudioCodecModel.restore_from', return_value=codec),
        patch('nemo.collections.tts.models.easy_magpietts_inference.setup_tokenizers', return_value=tokenizer),
        pytest.raises(ValueError, match='trust_remote_code'),
    ):
        EasyMagpieTTSInferenceModel(cfg)


@pytest.mark.unit
@pytest.mark.parametrize('model_name', ['MagpieTTSModel', 'EasyMagpieTTSInferenceModel'])
def test_tts_restore_rejects_custom_tokenizer_code(tmp_path, custom_tokenizer_repository, model_name):
    custom_repository = custom_tokenizer_repository
    from nemo.collections.tts.models.easy_magpietts_inference import EasyMagpieTTSInferenceModel
    from nemo.collections.tts.models.magpietts import MagpieTTSModel

    model_cls = {'MagpieTTSModel': MagpieTTSModel, 'EasyMagpieTTSInferenceModel': EasyMagpieTTSInferenceModel}[
        model_name
    ]
    config = OmegaConf.create(
        {
            'target': f'{model_cls.__module__}.{model_cls.__name__}',
            'codecmodel_path': 'unused.nemo',
            'text_tokenizers': {'text': {'_target_': 'AutoTokenizer', 'pretrained_model': custom_repository}},
        }
    )
    OmegaConf.save(config, tmp_path / 'model_config.yaml')
    torch.save({}, tmp_path / 'model_weights.ckpt')
    archive = tmp_path / 'model.nemo'
    with tarfile.open(archive, 'w') as dest:
        for name in ['model_config.yaml', 'model_weights.ckpt']:
            dest.add(tmp_path / name, arcname=name)
    with patch(f'{model_cls.__module__}.AudioCodecModel') as codec_cls:
        codec_cls.restore_from.side_effect = lambda *args, **kwargs: _mock_codec()
        codec_cls.from_pretrained.side_effect = lambda *args, **kwargs: _mock_codec()
        with pytest.raises(ValueError, match='trust_remote_code'):
            model_cls.restore_from(archive, map_location='cpu')


@pytest.fixture
def custom_tokenizer_repository(custom_repository):
    # A config without a built-in tokenizer isolates the custom-tokenizer branch.
    (Path(custom_repository) / 'config.json').write_text(json.dumps({'model_type': 'clip_vision_model'}))
    (Path(custom_repository) / 'tokenizer_config.json').write_text(
        json.dumps(
            {
                'auto_map': {'AutoTokenizer': ['tokenization_custom.CustomTokenizer', None]},
            }
        )
    )
    return custom_repository


def _mock_codec():
    codec = MagicMock()
    codec.sample_rate = codec.output_sample_rate = 22050
    codec.samples_per_frame = 1
    codec.num_codebooks = 1
    codec.codebook_size = 8
    return codec
