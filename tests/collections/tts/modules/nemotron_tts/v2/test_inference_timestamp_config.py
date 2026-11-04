# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from nemo.collections.tts.modules.magpietts_inference.inference import (
    EasyMagpieInferenceConfig,
    EasyMagpieInferenceRunner,
)

pytestmark = pytest.mark.unit


def test_reusing_a_model_resets_recording_to_each_runner_configuration(monkeypatch):
    monkeypatch.setattr(EasyMagpieInferenceRunner, '_configure_tokenizer', lambda self: None)
    model = SimpleNamespace(timestamp_enabled=False)
    model.configure_timestamps = lambda *, enabled: setattr(model, 'timestamp_enabled', enabled)
    EasyMagpieInferenceRunner(model, EasyMagpieInferenceConfig(save_timestamps=True))
    assert model.timestamp_enabled
    EasyMagpieInferenceRunner(model, EasyMagpieInferenceConfig(save_timestamps=False))
    assert not model.timestamp_enabled


def test_legacy_models_keep_working_when_timestamp_recording_is_disabled(monkeypatch):
    monkeypatch.setattr(EasyMagpieInferenceRunner, '_configure_tokenizer', lambda self: None)
    model = SimpleNamespace()
    assert EasyMagpieInferenceRunner(model, EasyMagpieInferenceConfig()).model is model
    with pytest.raises(ValueError, match='V2 timestamp API'):
        EasyMagpieInferenceRunner(model, EasyMagpieInferenceConfig(save_timestamps=True))
