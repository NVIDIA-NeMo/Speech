# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from nemo.collections.speechlm2.vllm.salm.config import NeMoSpeechLMConfig


@pytest.mark.unit
def test_salm_backbone_rejects_custom_repository_code(tmp_path):
    (tmp_path / 'config.json').write_text(
        json.dumps(
            {
                'model_type': 'custom_backend',
                'auto_map': {'AutoConfig': 'configuration_custom.CustomConfig'},
            }
        )
    )
    (tmp_path / 'configuration_custom.py').write_text("raise RuntimeError('unexpected module import')\n")
    with pytest.raises(ValueError, match='trust_remote_code'):
        NeMoSpeechLMConfig(
            pretrained_llm=str(tmp_path),
            pretrained_asr='unused',
            audio_locator_tag='<|audio|>',
            prompt_format='unused',
            pretrained_weights=False,
            trust_remote_code=True,
        )


@pytest.mark.unit
@pytest.mark.parametrize('setting', ['true', '1'])
def test_salm_vetted_backbone_can_be_enabled(tmp_path, monkeypatch, setting):
    monkeypatch.setenv('NEMO_TRUST_REMOTE_CODE', setting)
    (tmp_path / 'config.json').write_text(
        json.dumps(
            {
                'model_type': 'custom_backend',
                'architectures': ['GPT2LMHeadModel'],
                'vocab_size': 32,
                'n_layer': 1,
                'n_embd': 8,
                'n_head': 1,
                'auto_map': {'AutoConfig': 'configuration_custom.CustomConfig'},
            }
        )
    )
    (tmp_path / 'configuration_custom.py').write_text(
        'from transformers import GPT2Config\nclass CustomConfig(GPT2Config):\n    model_type = "custom_backend"\n'
    )
    config = NeMoSpeechLMConfig(
        pretrained_llm=str(tmp_path),
        pretrained_asr='unused',
        audio_locator_tag='<|audio|>',
        prompt_format='unused',
        pretrained_weights=False,
    )
    assert type(config.text_config).__name__ == 'CustomConfig'
    assert config.llm_architectures == ['GPT2LMHeadModel']


@pytest.fixture(autouse=True)
def default_remote_code_policy(monkeypatch):
    monkeypatch.delenv('NEMO_TRUST_REMOTE_CODE', raising=False)


@pytest.mark.unit
@pytest.mark.parametrize(
    'model_type,architecture',
    [
        ('qwen2', 'Qwen2ForCausalLM'),
        ('nemotron_h', 'NemotronHybridForCausalLM'),
    ],
)
def test_salm_uses_builtin_backbone_without_repository_code(tmp_path, model_type, architecture):
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    metadata = {
        'model_type': model_type,
        'architectures': [architecture],
        'hidden_size': 8,
        'intermediate_size': 16,
        'num_hidden_layers': 1,
        'num_attention_heads': 2,
        'num_key_value_heads': 2,
        'vocab_size': 32,
        'layer_norm_epsilon': 1e-5,
        'hybrid_override_pattern': '*',
        'auto_map': {'AutoConfig': 'configuration_custom.CustomConfig'},
    }
    (tmp_path / 'config.json').write_text(json.dumps(metadata))
    (tmp_path / 'configuration_custom.py').write_text("raise RuntimeError('unexpected module import')\n")
    config = NeMoSpeechLMConfig(
        pretrained_llm=str(tmp_path),
        pretrained_asr='unused',
        audio_locator_tag='<|audio|>',
        prompt_format='unused',
        pretrained_weights=False,
        trust_remote_code=True,
    )
    assert type(config.text_config) is CONFIG_MAPPING[model_type]
    assert config.text_config.hidden_size == 8
    assert config.text_config.speechlm_output_vocab_size == 32
    assert config.text_config.vocab_size == 42
    if model_type == 'nemotron_h':
        assert config.is_hybrid is True
        assert config.llm_architectures == ['NemotronHForCausalLM']
        assert config.text_config.total_num_kv_heads == 2
        assert config.text_config.rms_norm_eps == 1e-5
    else:
        assert config.is_hybrid is False
        assert config.llm_architectures == ['Qwen2ForCausalLM']
        assert config.text_config.layer_types == ['attention']


@pytest.mark.unit
def test_salm_uses_application_registered_config(tmp_path):
    from unittest.mock import patch

    from transformers import AutoConfig, PretrainedConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    class ApplicationConfig(PretrainedConfig):
        model_type = 'application_test'

    (tmp_path / 'config.json').write_text(
        json.dumps(
            {
                'model_type': ApplicationConfig.model_type,
                'architectures': ['ApplicationForCausalLM'],
                'vocab_size': 32,
                'num_hidden_layers': 1,
                'auto_map': {'AutoConfig': 'configuration_custom.CustomConfig'},
            }
        )
    )
    (tmp_path / 'configuration_custom.py').write_text("raise RuntimeError('unexpected module import')\n")
    with patch.dict(CONFIG_MAPPING._extra_content):
        AutoConfig.register(ApplicationConfig.model_type, ApplicationConfig)
        config = NeMoSpeechLMConfig(
            pretrained_llm=str(tmp_path),
            pretrained_asr='unused',
            audio_locator_tag='<|audio|>',
            prompt_format='unused',
            pretrained_weights=False,
            trust_remote_code=True,
        )
    assert isinstance(config.text_config, ApplicationConfig)
    assert config.llm_architectures == ['ApplicationForCausalLM']
    assert config.text_config.layer_types == ['attention']
