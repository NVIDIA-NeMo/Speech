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
